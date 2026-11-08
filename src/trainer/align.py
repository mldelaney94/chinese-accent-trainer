"""Forced alignment of a known transcript to audio with Qwen3-ForcedAligner (MLX).

What is aligned to what: AUDIO to TEXT. Pinyin is not a party to this; it inherits character
timings downstream because one character is one syllable.

The aligner tokenises Chinese text itself: each CJK character is one token; runs of Latin
letters/digits are one token; everything else (punctuation, whitespace) is dropped. We mirror
that tokenisation here so each returned timestamp can be mapped back to character indices in
the original transcript, and we assert the two token streams agree.

Failure modes this module detects (it cannot fix them):
- text with no audio  -> characters squeezed to implausibly short durations
- audio with no text  -> characters stretched to implausibly long durations, or a long
                         uncovered head/tail of audio
- model collapse past its reliable window -> non-monotonic or repeated timestamps
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from . import config

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from mlx_audio.stt.utils import load_model

        _MODEL = load_model(config.ALIGNER_MODEL)
    return _MODEL


# --- tokenisation mirror -------------------------------------------------------------------

def _is_cjk(ch: str) -> bool:
    c = ord(ch)
    return (
        0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF or 0x20000 <= c <= 0x2A6DF
        or 0x2A700 <= c <= 0x2B73F or 0x2B740 <= c <= 0x2B81F or 0x2B820 <= c <= 0x2CEAF
        or 0xF900 <= c <= 0xFAFF
    )


def _is_kept(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # char index in the joined text, inclusive
    end: int    # exclusive


def tokenize(text: str) -> list[Token]:
    """Mirror of mlx_audio ForceAlignProcessor.tokenize_chinese_mixed, with char spans."""
    tokens: list[Token] = []
    run_start: Optional[int] = None
    run: list[str] = []

    def flush(end: int) -> None:
        nonlocal run_start, run
        if run:
            tokens.append(Token("".join(run), run_start, end))
        run, run_start = [], None

    for i, ch in enumerate(text):
        if _is_cjk(ch):
            flush(i)
            tokens.append(Token(ch, i, i + 1))
        elif _is_kept(ch):
            if run_start is None:
                run_start = i
            run.append(ch)
        else:
            flush(i)
    flush(len(text))
    return tokens


# --- alignment -----------------------------------------------------------------------------

@dataclass
class Span:
    start: float
    end: float


def _align_window(audio: np.ndarray, text: str) -> list[tuple[str, float, float]]:
    res = _model().generate(audio=audio, text=text, language="Chinese")
    return [(it.text, float(it.start_time), float(it.end_time)) for it in res]


def align(audio: np.ndarray, text: str, chars_per_s: float = 4.5) -> list[Optional[Span]]:
    """Return one Span per token of `tokenize(text)`, in order.

    For audio longer than the aligner's reliable window, align iteratively: feed a window of
    audio and roughly the amount of text expected to fit it, trust only timestamps in the
    first ALIGN_TRUST_S seconds, then advance.
    """
    sr = config.SAMPLE_RATE
    tokens = tokenize(text)
    total_s = len(audio) / sr
    spans: list[Optional[Span]] = [None] * len(tokens)

    if total_s <= config.ALIGN_WINDOW_S:
        items = _align_window(audio, text)
        _check_tokens(tokens, items)
        for i, (_, s, e) in enumerate(items):
            spans[i] = Span(s, e)
        return spans

    tok_i = 0
    offset_s = 0.0
    while tok_i < len(tokens):
        remaining_s = total_s - offset_s
        win_s = min(config.ALIGN_WINDOW_S, remaining_s)
        a0 = int(offset_s * sr)
        a1 = int((offset_s + win_s) * sr)
        window = audio[a0:a1]

        if remaining_s <= config.ALIGN_WINDOW_S:
            sub_tokens = tokens[tok_i:]
        else:
            n_fit = int(win_s * chars_per_s * 1.2)
            sub_tokens = tokens[tok_i: tok_i + max(n_fit, 1)]
        sub_text = text[sub_tokens[0].start: sub_tokens[-1].end]
        items = _align_window(window, sub_text)
        _check_tokens(sub_tokens, items)

        final = remaining_s <= config.ALIGN_WINDOW_S
        trusted = 0
        last_end = 0.0
        for (_, s, e) in items:
            if not final and e > config.ALIGN_TRUST_S:
                break
            if s < last_end - 1e-6:  # non-monotonic: collapse begun
                break
            trusted += 1
            last_end = e
        if trusted == 0:
            raise RuntimeError(f"aligner trusted no tokens in window at {offset_s:.1f}s")
        for k in range(trusted):
            _, s, e = items[k]
            spans[tok_i + k] = Span(s + offset_s, e + offset_s)
        if final:
            break
        # re-estimate speaking rate from what we just trusted
        first_s = items[0][1]
        if last_end - first_s > 10:
            chars_per_s = trusted / (last_end - first_s)
        tok_i += trusted
        offset_s += last_end
    return spans


def _check_tokens(tokens: list[Token], items: list[tuple[str, float, float]]) -> None:
    if len(tokens) != len(items):
        raise RuntimeError(f"token count mismatch: ours={len(tokens)} aligner={len(items)}")
    for t, (txt, _, _) in zip(tokens, items):
        if t.text != txt:
            raise RuntimeError(f"token mismatch: ours={t.text!r} aligner={txt!r}")


# --- assembly + QC -------------------------------------------------------------------------

def transcript_coverage(paragraphs: list[str], audio_s: float) -> float:
    """Rough fraction of the audio the transcript could account for, from speaking rate."""
    n_han = sum(1 for ch in "".join(paragraphs) if _is_cjk(ch))
    return n_han / max(1.0, audio_s * config.EXPECTED_CHARS_PER_S)


class TranscriptCoverageError(RuntimeError):
    pass


def build_chars(paragraphs: list[str], audio: np.ndarray) -> tuple[list[dict[str, Any]], list[str]]:
    """Align the paragraphs to the audio and return (chars, qc_flags).

    chars: one dict per character of the joined transcript:
        {"i": global index, "p": paragraph index, "ch": char, "start": s|None, "end": s|None}
    Unaligned characters (punctuation) carry None times; the UI treats them as inert.

    Raises TranscriptCoverageError when the 文稿 is too short to be the spoken text.
    """
    audio_s = len(audio) / config.SAMPLE_RATE
    cov = transcript_coverage(paragraphs, audio_s)
    if cov < config.MIN_TRANSCRIPT_COVERAGE:
        raise TranscriptCoverageError(
            f"transcript covers ~{cov:.0%} of the audio by speaking rate; "
            f"the 文稿 is a summary, not the spoken text (interview-heavy segment?)"
        )
    text = "".join(paragraphs)
    para_of: list[int] = []
    for p_idx, p in enumerate(paragraphs):
        para_of.extend([p_idx] * len(p))

    spans = align(audio, text)
    tokens = tokenize(text)

    chars: list[dict[str, Any]] = [
        {"i": i, "p": para_of[i], "ch": ch, "start": None, "end": None} for i, ch in enumerate(text)
    ]
    for tok, sp in zip(tokens, spans):
        if sp is None:
            continue
        for k in range(tok.start, tok.end):
            chars[k]["start"], chars[k]["end"] = round(sp.start, 3), round(sp.end, 3)

    flags = qc(chars, spans, tokens, audio_s)
    if cov < 0.8:
        flags.append(f"transcript covers only ~{cov:.0%} of the audio by speaking rate")
    return chars, flags


def qc(chars: list[dict[str, Any]], spans: list[Optional[Span]], tokens: list[Token], audio_s: float) -> list[str]:
    flags: list[str] = []
    cjk = [(t, s) for t, s in zip(tokens, spans) if s is not None and len(t.text) == 1 and _is_cjk(t.text)]
    if not cjk:
        return ["no aligned characters"]
    durs = np.array([s.end - s.start for _, s in cjk])
    starts = np.array([s.start for _, s in cjk])

    too_short = int((durs < config.SYLLABLE_MIN_S).sum())
    too_long = int((durs > config.SYLLABLE_MAX_S).sum())
    if too_short:
        flags.append(f"{too_short} syllables shorter than {config.SYLLABLE_MIN_S}s (text with no audio?)")
    if too_long:
        worst = max(cjk, key=lambda ts: ts[1].end - ts[1].start)
        flags.append(
            f"{too_long} syllables longer than {config.SYLLABLE_MAX_S}s (audio with no text?) "
            f"e.g. {worst[0].text!r} {worst[1].start:.2f}-{worst[1].end:.2f}"
        )
    if np.any(np.diff(starts) < -1e-6):
        flags.append("non-monotonic timestamps (aligner collapse?)")
    head = float(starts[0])
    tail = audio_s - float(cjk[-1][1].end)
    if head > 3.0:
        flags.append(f"{head:.1f}s of audio before first character")
    if tail > 5.0:
        flags.append(f"{tail:.1f}s of audio after last character")
    missing = sum(1 for s in spans if s is None)
    if missing:
        flags.append(f"{missing} tokens unaligned")
    return flags
