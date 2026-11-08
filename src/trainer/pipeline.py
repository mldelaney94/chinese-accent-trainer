"""Orchestration: one segment URL (or one broadcast date) through every stage.

Artefact per segment: data/segments/{guid}.json
{
  guid, slug, page_url, headline, broadcast_date, duration_s, cctv_tags,
  paragraphs: [str],            # transcript, byline stripped
  english:    [str] | null,     # one per paragraph
  chars:      [{i, p, ch, py, start, end}],
  words:      [{p, s, e, w}],
  qc_flags:   [str]
}
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import soundfile as sf

from . import align, cctv, config, db, enrich


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _date_from(pgm_time: Optional[str], page_url: str) -> str:
    if pgm_time:
        return pgm_time[:10]
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", page_url)
    if m:
        return "-".join(m.groups())
    return date.today().isoformat()


def segment_json_path(guid: str) -> Path:
    return config.SEGMENT_DIR / f"{guid}.json"


def load_segment_json(guid: str) -> Optional[dict[str, Any]]:
    p = segment_json_path(guid)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def process_url(url: str, translate: bool = True, force: bool = False) -> str:
    """Run every stage for one segment page. Returns the GUID."""
    config.ensure_dirs()
    db.init_db()

    page = cctv.fetch_segment_page(url)
    guid = page.guid
    existing = db.get_segment(guid)
    if existing and existing.status == "ready" and not force:
        _log(f"[{guid}] already ready: {existing.headline}")
        return guid
    if not page.paragraphs:
        db.upsert_segment(guid, _date_from(None, url), page.headline, page.slug, url)
        db.set_status(guid, "failed", "no transcript on page")
        raise RuntimeError(f"[{guid}] page has no 文稿: {url}")

    media = cctv.resolve_media(guid, url)
    bdate = _date_from(media.pgm_time, url)
    db.upsert_segment(guid, bdate, page.headline, page.slug, url, media.duration_s, media.tags)
    _log(f"[{guid}] {bdate} {page.headline} ({media.duration_s}s, {len(page.paragraphs)} paras)")

    try:
        wav = config.AUDIO_DIR / f"{guid}.wav"
        if not wav.exists():  # audio is immutable content; --force re-processes but never re-downloads
            _log(f"[{guid}] downloading audio")
            cctv.download_audio(media.mp4_urls, wav)
        db.set_status(guid, "audio")

        audio, sr = sf.read(wav, dtype="float32")
        assert sr == config.SAMPLE_RATE, sr
        _log(f"[{guid}] aligning {len(audio)/sr:.1f}s")
        try:
            chars, flags = align.build_chars(page.paragraphs, audio)
        except align.TranscriptCoverageError as e:
            db.set_status(guid, "unusable", str(e), qc_flags="[]")
            _log(f"[{guid}] UNUSABLE: {e}")
            return guid
        db.set_status(guid, "aligned", qc_flags=json.dumps(flags, ensure_ascii=False))
        for f in flags:
            _log(f"[{guid}]   QC: {f}")

        extra = (media.tags or "").split() + re.split(r"[\s　【】：:]+", page.headline)
        chars, words = enrich.enrich(page.paragraphs, chars, extra_words=extra)
        db.set_status(guid, "enriched")

        english: Optional[list[str]] = None
        if translate:
            from . import translate as tr

            _log(f"[{guid}] translating {len(page.paragraphs)} paragraphs")
            english = tr.translate_paragraphs(page.paragraphs)
            db.set_status(guid, "translated")

        doc = {
            "guid": guid,
            "slug": page.slug,
            "page_url": url,
            "headline": page.headline,
            "broadcast_date": bdate,
            "duration_s": media.duration_s or round(len(audio) / sr, 2),
            "cctv_tags": media.tags,
            "paragraphs": page.paragraphs,
            "english": english,
            "chars": chars,
            "words": words,
            "qc_flags": flags,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }
        segment_json_path(guid).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        db.set_status(guid, "ready" if (english is not None or not translate) else "enriched")
        _log(f"[{guid}] ready")
        return guid
    except Exception as e:  # record, then re-raise so the caller sees it
        db.set_status(guid, "failed", f"{type(e).__name__}: {e}")
        raise


def translate_existing(guid: str, force: bool = False) -> bool:
    """Add English to an already-processed segment. Returns True if translation ran."""
    doc = load_segment_json(guid)
    if doc is None:
        raise FileNotFoundError(f"no segment json for {guid}")
    if doc.get("english") and not force:
        return False
    from . import translate as tr

    _log(f"[{guid}] translating {len(doc['paragraphs'])} paragraphs")
    doc["english"] = tr.translate_paragraphs(doc["paragraphs"])
    segment_json_path(guid).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    db.set_status(guid, "ready")
    return True


def process_day(d: date, translate: bool = True, force: bool = False, include_full: bool = False) -> list[str]:
    """Ingest every story segment for one broadcast date. Skips the 完整版 unless asked."""
    items = cctv.fetch_day(d)
    guids: list[str] = []
    for it in items:
        if it.is_full_episode and not include_full:
            continue
        try:
            guids.append(process_url(it.page_url, translate=translate, force=force))
        except Exception as e:
            _log(f"  FAILED {it.title}: {e}")
    return guids
