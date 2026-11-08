"""CCTV 新闻联播 ingestion.

Verified against the live site (2026-11):
- Daily listing is a static page:  https://tv.cctv.com/lm/xwlb/day/YYYYMMDD.shtml
- Segment pages are server-rendered. The content key is `var guid = "<32 hex>"` in the page
  source. The VIDE... slug in the URL is a page identifier, not the content identifier.
- The 文稿 (transcript) is in <div id="content_area"> as <p> paragraphs. The first paragraph
  carries an editorial byline 央视网消息（新闻联播）： which is NOT spoken and must be stripped.
- Stream info: https://vdn.apps.cntv.cn/api/getHttpVideoInfo.do?pid=<guid>
  `video.chapters4[0].url` is a plain MP4 with 128 kbps AAC; chapters/chapters2 are 32 kbps.
"""

from __future__ import annotations

import html
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from . import config

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
DAY_URL = "https://tv.cctv.com/lm/xwlb/day/{ymd}.shtml"
VDN_URL = "https://vdn.apps.cntv.cn/api/getHttpVideoInfo.do"

GUID_RE = re.compile(r'var\s+guid\s*=\s*"([0-9a-f]{32})"')
SLUG_RE = re.compile(r"/(VIDE[0-9A-Za-z]+)\.shtml")
BYLINE_RE = re.compile(r"^\s*央视网消息\s*[（(][^）)]*[）)]\s*[：:]\s*")
FULL_EPISODE_RE = re.compile(r"^《新闻联播》\s*\d{8}")


@dataclass
class Listing:
    page_url: str
    title: str
    duration_s: Optional[float]

    @property
    def is_full_episode(self) -> bool:
        return bool(FULL_EPISODE_RE.match(self.title))


@dataclass
class SegmentPage:
    guid: str
    slug: Optional[str]
    page_url: str
    headline: str
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class MediaInfo:
    mp4_urls: list[str]          # CCTV splits long videos into ~240 s chapter files
    duration_s: Optional[float]
    pgm_time: Optional[str]
    tags: Optional[str]
    title: Optional[str]


def _get(url: str, retries: int = 3, **kw) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(url, headers=UA, follow_redirects=True, timeout=30, **kw)
            r.raise_for_status()
            return r
        except (httpx.TransportError, httpx.HTTPStatusError) as e:  # timeouts, resets, 5xx
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts") from last


def _hms_to_s(s: str) -> Optional[float]:
    parts = s.strip().split(":")
    if not all(p.isdigit() for p in parts):
        return None
    parts = [int(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return float(h * 3600 + m * 60 + sec)


def fetch_day(d: date) -> list[Listing]:
    """All items for one broadcast date, in page order. Includes the 完整版 full episode."""
    r = _get(DAY_URL.format(ymd=d.strftime("%Y%m%d")))
    tree = HTMLParser(r.text)
    seen: dict[str, Listing] = {}
    for li in tree.css("li"):
        a = li.css_first("a[href*='VIDE']")
        if a is None:
            continue
        href = a.attributes.get("href", "")
        title = a.attributes.get("title") or a.text(strip=True)
        if not href or href in seen:
            continue
        span = li.css_first("span")
        dur = _hms_to_s(span.text()) if span is not None else None
        seen[href] = Listing(page_url=href, title=html.unescape(title), duration_s=dur)
    return list(seen.values())


def fetch_segment_page(url: str) -> SegmentPage:
    r = _get(url)
    src = r.text
    m = GUID_RE.search(src)
    if not m:
        raise ValueError(f"no guid in page {url}")
    guid = m.group(1)
    slug_m = SLUG_RE.search(url)
    tree = HTMLParser(src)
    title_node = tree.css_first("title")
    headline = html.unescape(title_node.text(strip=True)) if title_node else url
    headline = re.sub(r"^\[视频\]", "", headline).strip()

    paragraphs: list[str] = []
    area = tree.css_first("#content_area")
    if area is not None:
        for p in area.css("p"):
            txt = html.unescape(p.text(strip=True))
            txt = re.sub(r"\s+", "", txt)
            if txt:
                paragraphs.append(txt)
    if paragraphs:
        paragraphs[0] = BYLINE_RE.sub("", paragraphs[0])
        paragraphs = [p for p in paragraphs if p]
    return SegmentPage(guid=guid, slug=slug_m.group(1) if slug_m else None, page_url=url,
                       headline=headline, paragraphs=paragraphs)


def resolve_media(guid: str, page_url: Optional[str] = None) -> MediaInfo:
    params = {"pid": guid, "idl": 32, "idlr": 32, "modifyed": "false"}
    if page_url:
        params["url"] = page_url
    r = _get(VDN_URL, params=params)
    d = r.json()
    video = d.get("video") or {}
    urls: list[str] = []
    for key in ("chapters4", "chapters3", "chapters2", "chapters"):
        ch = video.get(key) or []
        urls = [c["url"] for c in ch if c.get("url")]
        if urls:
            break
    if not urls:
        raise ValueError(f"no mp4 url for guid {guid}: keys={list(video.keys())}")
    dur = video.get("totalLength")
    return MediaInfo(
        mp4_urls=urls,
        duration_s=float(dur) if dur else None,
        pgm_time=d.get("f_pgmtime"),
        tags=d.get("tag") or None,
        title=d.get("title"),
    )


def _download_file(url: str, dest: Path, max_attempts: int = 8) -> None:
    """Resumable download. The CCTV CDN closes connections mid-body routinely."""
    have = dest.stat().st_size if dest.exists() else 0
    total: Optional[int] = None
    for attempt in range(max_attempts):
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=120) as r:
                if r.status_code == 416:  # range not satisfiable: already complete
                    return
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored Range: start over
                    have = 0
                if total is None:
                    cl = r.headers.get("content-length")
                    if cl:
                        total = have + int(cl)
                with open(dest, "ab" if have else "wb") as f:
                    for chunk in r.iter_bytes(1 << 16):
                        f.write(chunk)
                        have += len(chunk)
            if total is None or have >= total:
                return
        except (httpx.TransportError, httpx.HTTPStatusError):
            pass
        time.sleep(min(10, 1.5 * (attempt + 1)))
        have = dest.stat().st_size if dest.exists() else 0
    raise RuntimeError(f"download failed after {max_attempts} attempts: {url} ({have}/{total} bytes)")


def download_audio(mp4_urls: list[str], out_wav: Path) -> Path:
    """Download every chapter file, concatenate, transcode to 16 kHz mono PCM WAV."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # .part files are kept on failure so a re-run resumes instead of restarting
    parts = [out_wav.with_suffix(f".{i}.mp4.part") for i in range(len(mp4_urls))]
    for url, part in zip(mp4_urls, parts):
        _download_file(url, part)
    if len(parts) == 1:
        inputs = ["-i", str(parts[0])]
        extra: list[Path] = []
    else:
        lst = out_wav.with_suffix(".concat.txt")
        lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
        inputs = ["-f", "concat", "-safe", "0", "-i", str(lst)]
        extra = [lst]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", *inputs,
        "-vn", "-ac", "1", "-ar", str(config.SAMPLE_RATE), "-c:a", "pcm_s16le", str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    for p in parts + extra:
        p.unlink(missing_ok=True)
    return out_wav
