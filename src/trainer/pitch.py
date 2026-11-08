"""Pitch contours and their comparison.

Two axes, two operations:
- PITCH axis: Hz -> semitones relative to the speaker's own median F0 over the utterance.
  Perception of pitch is logarithmic in frequency; the semitone is the log unit (12 per octave).
  Subtracting each speaker's median discards register (she is ~200 Hz, you are ~110 Hz) and
  leaves shape, which is what tone is.
- TIME axis: dynamic time warping finds the monotonic path through the cost matrix that best
  lines up two contours of unequal length. It will warp anything onto anything; the residual
  cost along the path is a distance, not a verdict.

The same tracker (Praat autocorrelation via parselmouth) is used on both sides. Different
trackers have different octave-error and voicing-decision behaviour; mixing them would put
tracker disagreement into the comparison and call it accent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import parselmouth

from . import config


@dataclass
class Contour:
    times: np.ndarray       # seconds, hop = PITCH_TIME_STEP
    f0_hz: np.ndarray       # NaN where unvoiced
    semitones: np.ndarray   # NaN where unvoiced; 0 == speaker median
    median_hz: float

    def to_json(self) -> dict[str, Any]:
        def clean(a: np.ndarray) -> list[Optional[float]]:
            return [None if np.isnan(v) else round(float(v), 3) for v in a]

        return {
            "times": [round(float(t), 3) for t in self.times],
            "f0_hz": clean(self.f0_hz),
            "semitones": clean(self.semitones),
            "median_hz": round(self.median_hz, 2),
        }


def extract_f0(samples: np.ndarray, sr: int, t0: float = 0.0, t1: Optional[float] = None) -> Contour:
    """F0 track for samples[t0:t1] (seconds). Times are absolute (offset by t0)."""
    if t1 is not None:
        samples = samples[int(t0 * sr): int(t1 * sr)]
    elif t0:
        samples = samples[int(t0 * sr):]
    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(
        time_step=config.PITCH_TIME_STEP,
        pitch_floor=config.PITCH_FLOOR_HZ,
        pitch_ceiling=config.PITCH_CEILING_HZ,
    )
    f0 = pitch.selected_array["frequency"].astype(np.float64)
    f0[f0 <= 0] = np.nan
    times = pitch.xs() + t0
    f0 = _clean_octave_errors(f0)
    voiced = f0[~np.isnan(f0)]
    median = float(np.median(voiced)) if voiced.size else float("nan")
    st = 12.0 * np.log2(f0 / median) if voiced.size else np.full_like(f0, np.nan)
    return Contour(times=times, f0_hz=f0, semitones=st, median_hz=median)


def _clean_octave_errors(f0: np.ndarray, max_semitones_from_median: float = 12.0,
                         max_jump_semitones: float = 6.0) -> np.ndarray:
    """Drop frames that are implausible for one speaker's connected speech:
    more than an octave from the utterance median, or a jump of more than half an octave
    between adjacent voiced frames (10 ms). Both are tracker halving/doubling, not pitch."""
    out = f0.copy()
    voiced = ~np.isnan(out)
    if voiced.sum() < 3:
        return out
    med = np.median(out[voiced])
    st = 12.0 * np.log2(out / med)
    out[np.abs(st) > max_semitones_from_median] = np.nan
    # adjacent-frame jumps: mark the frame that departs from its neighbour
    for i in range(1, out.size):
        if np.isnan(out[i]) or np.isnan(out[i - 1]):
            continue
        if abs(12.0 * np.log2(out[i] / out[i - 1])) > max_jump_semitones:
            out[i] = np.nan
    return out


def _fill_unvoiced(st: np.ndarray) -> np.ndarray:
    """Linear interpolation across unvoiced gaps so DTW has a value everywhere.
    Voicing mask is kept separately by the caller for display."""
    x = np.arange(st.size)
    ok = ~np.isnan(st)
    if ok.sum() < 2:
        return np.zeros_like(st)
    return np.interp(x, x[ok], st[ok])


def dtw(a: np.ndarray, b: np.ndarray, band: Optional[int] = None) -> tuple[list[tuple[int, int]], float]:
    """Classic DTW with |i-j| <= band (Sakoe-Chiba). Returns (path, normalised cost).

    a, b: 1-D sequences (semitones, unvoiced filled). Cost is absolute difference.
    """
    n, m = a.size, b.size
    if band is None:
        band = max(n, m) // 4 + 1
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtrack
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        steps = [(D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1)]
        _, i, j = min(steps)
        path.append((i - 1, j - 1))
    path.reverse()
    return path, float(D[n, m] / len(path))


def compare(ref: Contour, usr: Contour) -> dict[str, Any]:
    """Warp the learner's contour onto the reference's time axis.

    Returns the reference contour, the learner's contour, the warp path, the learner's
    semitones resampled onto reference frames (for overlay), and the mean absolute
    semitone difference over frames where both are voiced.
    """
    a = _fill_unvoiced(ref.semitones)
    b = _fill_unvoiced(usr.semitones)
    path, cost = dtw(a, b)

    # for each reference frame, the learner value(s) mapped to it (mean if several)
    mapped = np.full(a.size, np.nan)
    counts = np.zeros(a.size)
    for i, j in path:
        v = usr.semitones[j]
        if not np.isnan(v):
            mapped[i] = (0.0 if np.isnan(mapped[i]) else mapped[i]) + v
            counts[i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mapped = np.where(counts > 0, mapped / np.maximum(counts, 1), np.nan)

    both = ~np.isnan(ref.semitones) & ~np.isnan(mapped)
    mad = float(np.mean(np.abs(ref.semitones[both] - mapped[both]))) if both.any() else None

    return {
        "ref": ref.to_json(),
        "usr": usr.to_json(),
        "path": path,
        "usr_on_ref": [None if np.isnan(v) else round(float(v), 3) for v in mapped],
        "dtw_cost": round(cost, 4),
        "mean_abs_semitone_diff": None if mad is None else round(mad, 3),
    }
