"""Paths and tunables. Everything lives under one data directory, addressed by segment GUID."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("TRAINER_DATA", "data")).resolve()
AUDIO_DIR = DATA_DIR / "audio"          # {guid}.wav   16 kHz mono PCM
SEGMENT_DIR = DATA_DIR / "segments"     # {guid}.json  transcript + alignment + pinyin + translation
PITCH_DIR = DATA_DIR / "pitch"          # {guid}.json  cached reference F0 contour
RECORDING_DIR = DATA_DIR / "recordings" # {id}.wav     learner recordings
DB_PATH = DATA_DIR / "trainer.sqlite"

ALIGNER_MODEL = os.environ.get("TRAINER_ALIGNER", "mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
TRANSLATOR_MODEL = os.environ.get("TRAINER_TRANSLATOR", "mlx-community/Qwen3-8B-4bit")

SAMPLE_RATE = 16_000

# Qwen3-ForcedAligner is trained for <= 300 s; reliable to ~270 s. Beyond that it collapses
# trailing timestamps. We align in windows and only trust the first part of each window.
ALIGN_WINDOW_S = 240.0
ALIGN_TRUST_S = 200.0

# Broadcast Mandarin syllables: median ~160 ms, p95 ~320 ms (measured). Outside this band
# the aligner is probably smearing text over audio that does not contain it (or vice versa).
SYLLABLE_MIN_S = 0.04
SYLLABLE_MAX_S = 1.0

# Broadcast speaking rate (measured ~4.4 Han chars/s on anchor-read copy). If the transcript
# has far fewer characters than the audio duration implies, the 文稿 is a summary and does not
# cover the speech (interview-heavy pieces). Alignment of such a segment is meaningless.
EXPECTED_CHARS_PER_S = 4.2
MIN_TRANSCRIPT_COVERAGE = 0.5

# Pitch tracking
PITCH_TIME_STEP = 0.01
PITCH_FLOOR_HZ = 60.0
PITCH_CEILING_HZ = 500.0


def ensure_dirs() -> None:
    for d in (AUDIO_DIR, SEGMENT_DIR, PITCH_DIR, RECORDING_DIR):
        d.mkdir(parents=True, exist_ok=True)
