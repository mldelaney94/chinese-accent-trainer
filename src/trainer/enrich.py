"""Pinyin (underlying tones) and word segmentation. Both are display-layer concerns.

- Segmentation: rjieba (Rust jieba), with character offsets. Publisher tags and headline
  terms are added to its dictionary so domain compounds (算力网) segment as one word.
- Pinyin: pypinyin, fed the SAME segmentation as a pre-segmented list so phrase context
  (which resolves most 多音字: 了解 -> liǎo, 还在 -> hái) comes from one source of truth.
- Tone sandhi: pypinyin applies none by default (it is an opt-in mixin). But its phrase
  dictionary stores sandhi'd readings for 一 and 不 (一个 -> yí gè, 不是 -> bú shì). This tool
  displays underlying tones and expects the learner to apply sandhi, so those two characters
  are reverted to yī / bù. Third-tone sandhi is not in the dictionary and needs no handling.
"""

from __future__ import annotations

from typing import Any

import rjieba
from pypinyin import Style, pinyin

_UNDERLYING = {"一": "yī", "不": "bù"}


def add_words(words: list[str]) -> None:
    for w in words:
        w = w.strip()
        if len(w) >= 2 and all("\u4e00" <= ch <= "\u9fff" for ch in w):
            rjieba.add_word(w, None, None)


def segment(paragraph: str) -> list[tuple[str, int, int]]:
    """(word, start, end) with character offsets into `paragraph`."""
    return [(w, s, e) for w, s, e in rjieba.tokenize(paragraph) if w.strip()]


def pinyin_for(paragraph: str, words: list[tuple[str, int, int]]) -> list[str]:
    """One pinyin string per character of `paragraph`; '' for non-Han characters."""
    out = pinyin([w for w, _, _ in words], style=Style.TONE,
                 errors=lambda chunk: [""] * len(chunk))
    result = [x[0] if x else "" for x in out]
    # rjieba drops nothing but whitespace; whitespace was already removed at ingest.
    assert len(result) == len(paragraph), (len(result), len(paragraph), paragraph[:30])
    for i, ch in enumerate(paragraph):
        if ch in _UNDERLYING and result[i]:
            result[i] = _UNDERLYING[ch]
    return result


def enrich(
    paragraphs: list[str], chars: list[dict[str, Any]], extra_words: list[str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach 'py' to each char dict; return (chars, words) where words carry global spans."""
    add_words(extra_words or [])
    words_out: list[dict[str, Any]] = []
    base = 0
    for p_idx, para in enumerate(paragraphs):
        words = segment(para)
        pys = pinyin_for(para, words)
        for k, py in enumerate(pys):
            chars[base + k]["py"] = py
        for w, s, e in words:
            words_out.append({"p": p_idx, "s": base + s, "e": base + e, "w": w})
        base += len(para)
    assert base == len(chars)
    return chars, words_out
