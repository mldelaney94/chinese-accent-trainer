"""Paragraph-level zh -> en translation with a local Qwen3 model via mlx-lm.

Paragraph granularity is free: the 文稿 is already paragraphed and the anchor pauses between
paragraphs, so the English panel scrolls in lockstep with the Chinese without any further
alignment. One call per paragraph keeps the mapping trivially 1:1.
"""

from __future__ import annotations

import re

from . import config

_MODEL = None
_TOK = None

SYSTEM = (
    "You are a professional news translator. Translate the Chinese news text into natural, "
    "accurate English. Preserve meaning, names, numbers and official titles exactly. "
    "Glossary: 十五五 = the 15th Five-Year Plan (2026-2030); 十四五 = the 14th Five-Year Plan; "
    "二十届五中全会 = the Fifth Plenary Session of the 20th CPC Central Committee; "
    "新闻联播 = Xinwen Lianbo; 东数西算 = East Data, West Computing. "
    "Output only the English translation, with no preamble, notes, or quotation marks."
)


def _load():
    global _MODEL, _TOK
    if _MODEL is None:
        from mlx_lm import load

        _MODEL, _TOK = load(config.TRANSLATOR_MODEL)
    return _MODEL, _TOK


def _strip_thinking(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    return s.strip().strip('"').strip()


def translate_paragraph(zh: str, max_tokens: int = 600) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tok = _load()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": zh}]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False,
                   sampler=make_sampler(temp=0.0))
    return _strip_thinking(out)


def translate_paragraphs(paragraphs: list[str]) -> list[str]:
    return [translate_paragraph(p) for p in paragraphs]
