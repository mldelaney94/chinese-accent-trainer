# chinese-accent-trainer

Shadowing trainer on CCTV 新闻联播. No ASR anywhere: the 文稿 supplies the text, forced
alignment supplies the timing, Praat + DTW supply the comparison. Everything local, MLX.

```sh
uv run trainer ingest 2026-10-31      # or a single VIDE… URL; --no-translate to skip the LLM
uv run trainer translate              # fill in English later
uv run trainer list                   # status + QC flags per segment
uv run trainer serve                  # http://127.0.0.1:8765
```

Data under `data/` (`TRAINER_DATA` relocates it): `trainer.sqlite`, `audio/{guid}.wav`,
`segments/{guid}.json`, `recordings/{id}.wav`. Key everywhere is CCTV's 32-hex content GUID.

## 1. Forced alignment — `src/trainer/align.py`

**Model:** `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`, loaded through `mlx_audio.stt.utils.load_model`,
called as `model.generate(audio=<16 kHz float32>, text=<joined 文稿>, language="Chinese")`.

- **Tokenisation.** The aligner tokenises Chinese itself: each CJK character = one token,
  runs of Latin/digits = one token, punctuation/whitespace dropped. `tokenize()` mirrors
  that (`tokenize_chinese_mixed` in mlx_audio) so each returned timestamp maps back to
  character indices in the original transcript; `_check_tokens` asserts the two streams match
  exactly and raises otherwise.
- **Resolution** is 80 ms frames. Broadcast syllables measured at median 160 ms, p95 320 ms —
  boundaries are good to a few tens of ms, not better.
- **Long audio.** Reliable to ~270 s. Over `ALIGN_WINDOW_S` (240 s) it runs iteratively:
  feed a 240 s audio window plus `win_s × chars_per_s × 1.2` tokens of text, trust only items
  ending before `ALIGN_TRUST_S` (200 s) and monotonic, re-estimate `chars_per_s` from what
  was trusted, advance the offset. Validated on a doubled 285 s input: ≤1 frame off the
  single-pass result, monotonic across the seam.
- **Pinyin is not aligned.** One character = one syllable, so `enrich.py` gives each
  character a pinyin (pypinyin `Style.TONE`, phrase context from rjieba words) and the pinyin
  inherits the character's `start/end`. Tones shown are underlying; pypinyin's phrase dict
  stores sandhi'd 一/不 (yí gè, bú shì) as data, not via a flag, so those two are reverted
  in `pinyin_for`. rjieba word spans are display grouping only; publisher `tag` terms and
  the headline are added to its dictionary first (`算力网` otherwise splits as 算力/网是).
- **QC** (`qc()`), written to `qc_flags`: syllables <40 ms (text with no audio), >1 s with
  the worst example (audio with no text), non-monotonic starts (collapse), >3 s head or
  >5 s tail uncovered, unaligned tokens. Before aligning, `transcript_coverage` =
  Han chars ÷ (audio_s × 4.2); <0.5 raises `TranscriptCoverageError` → segment `unusable`
  (the 文稿 summarises interview soundbites rather than transcribing them); <0.8 is flagged.

## 2. CCTV ingestion — `src/trainer/cctv.py`, `pipeline.py`

All verified against the live site, Nov 2026.

| step | source | how |
|---|---|---|
| daily listing | `https://tv.cctv.com/lm/xwlb/day/YYYYMMDD.shtml` | static HTML; `li > a[href*='VIDE']`, `title` attr, `span` duration. 完整版 detected by `^《新闻联播》\d{8}` and skipped unless `--include-full` |
| content key | segment page source | `var guid = "<32 hex>"` — the `VIDE…` slug in the URL is a page label, not the content id |
| transcript | `<div id="content_area"> p` | lexbor parse (`selectolax.lexbor`); whitespace collapsed; first paragraph's editorial byline `央视网消息（新闻联播）：` stripped (not spoken) |
| headline | `<title>` minus `[视频]` prefix | |
| stream info | `https://vdn.apps.cntv.cn/api/getHttpVideoInfo.do?pid=GUID&idl=32&idlr=32&modifyed=false&url=…` | JSON; takes every `url` in `video.chapters4` (128 kbps AAC), falling back chapters3→2→chapters (32 kbps). **Long videos are split into ~240 s chapter files** — all are downloaded and concatenated, otherwise you align half the audio. Also returns `totalLength`, `f_pgmtime`, `tag` (used as segmenter hints) |
| download | CDN | `_download_file`: resumable `Range: bytes=N-`, handles 206/416, 8 attempts, `.part` kept on failure. The CDN closes connections mid-body routinely |
| transcode | ffmpeg | concat demuxer if >1 chapter → `-vn -ac 1 -ar 16000 -c:a pcm_s16le` |

`pipeline.process_url`: page → `upsert_segment` → `resolve_media` → download (only if the WAV
is missing; `--force` re-aligns, never re-downloads) → `build_chars` → `enrich` → optional
translate → `data/segments/{guid}.json` with `paragraphs, english, chars[{i,p,ch,start,end,py}],
words[{p,s,e,w}], qc_flags`. Status column: `new→audio→aligned→enriched→translated→ready`,
or `unusable` / `failed` with `error`.

Translation (`translate.py`) is the only LLM: `mlx-community/Qwen3-8B-4bit` via mlx-lm,
`enable_thinking=False`, temp 0, one paragraph per call, glossary pinned in the system
prompt (十五五 was hallucinated as "14th Five-Year Plan" without it). Extend the glossary
as you read.

## 3. Comparing my speech to the anchor's — `src/trainer/pitch.py`, `server.py`, `app/`

**Library:** Praat via `praat-parselmouth`. DTW is hand-written (`dtw()`), no library.

Preprocessing, identical on both sides so tracker behaviour cancels out:

1. **Audio in.** Reference = `audio/{guid}.wav[t0:t1]` for the selected paragraph (character
   timings from alignment give `t0,t1`). Mine = browser `MediaRecorder` (webm/opus or mp4)
   POSTed to `/api/segments/{guid}/compare`, ffmpeg'd to 16 kHz mono PCM into
   `data/recordings/`.
2. **F0 track.** `parselmouth.Sound(...).to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=500)`
   — Praat autocorrelation, 10 ms hop. Unvoiced frames → NaN.
3. **Octave-error cleanup** (`_clean_octave_errors`): drop frames >12 st from the utterance
   median or >6 st from the previous voiced frame. Those are tracker halving/doubling, not
   pitch; without this the contour showed −25 st plunges.
4. **Semitones re. own median:** `12·log2(f0 / median_f0_of_speaker)`. Discards register (her
   ~200 Hz vs my ~110 Hz), keeps shape, which is what tone is. Verified: anchor shifted
   −8.5 st compares at 0.45 st mean |Δ|; identity at 0.0.
5. **Fill unvoiced** by linear interpolation (`_fill_unvoiced`) so DTW has a value at every
   frame; the voicing mask is kept separately for display.
6. **DTW** on the two semitone sequences, absolute-difference cost, Sakoe-Chiba band
   `max(n,m)//4 + 1`, path normalised by its length. Returns the path, my contour resampled
   onto the reference's frames (`usr_on_ref`, mean where several frames map to one), the
   normalised `dtw_cost`, and `mean_abs_semitone_diff` over frames voiced in both.

The UI (`app/app.js`) draws reference (blue), my warped contour (red), and the live
unprocessed contour from `pitch-worklet.js` (grey; 40 ms window / 10 ms hop normalised
autocorrelation in an AudioWorklet, 60–500 Hz lags) on a ±12 st grid with character
boundaries from the alignment. The numbers are distances, not grades: DTW will warp anything
onto anything. Read the overlay.

## Things that will bite

- A segment's 文稿 can be a summary (interview pieces) — check `qc_flags` and `status` before
  trusting the highlighting.
- 多音字 are resolved by phrase context, not always correctly. Check pinyin by ear.
- Highlighting polls `audio.currentTime` via rAF with binary search over `chars`, not
  `timeupdate` (which fires ~4 Hz).
- Models (~1.4 GB aligner, ~4.6 GB translator) download from Hugging Face on first use.
  Needs `brew install uv ffmpeg`, `uv sync`.
