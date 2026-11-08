"""Local web app: FastAPI serving the segment API, audio, pitch endpoints, and the static UI."""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db, pipeline, pitch

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"

app = FastAPI(title="chinese-accent-trainer")
db.init_db()
config.ensure_dirs()


@app.middleware("http")
async def no_cache(request, call_next):
    # segments are re-processed in place; the browser must never serve a stale copy
    resp = await call_next(request)
    if request.url.path.startswith("/api/") and not request.url.path.endswith("/audio"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/segments")
def api_segments(date_from: Optional[str] = None, date_to: Optional[str] = None):
    segs = db.list_segments(
        date.fromisoformat(date_from) if date_from else None,
        date.fromisoformat(date_to) if date_to else None,
    )
    return [
        {
            "guid": s.guid, "date": s.broadcast_date, "headline": s.headline,
            "duration_s": s.duration_s, "status": s.status, "tags": s.cctv_tags,
            "qc_flags": json.loads(s.qc_flags) if s.qc_flags else [],
            "error": s.error,
        }
        for s in segs
    ]


@app.get("/api/segments/{guid}")
def api_segment(guid: str):
    doc = pipeline.load_segment_json(guid)
    if doc is None:
        raise HTTPException(404, "segment not processed")
    return doc


@app.get("/api/segments/{guid}/audio")
def api_audio(guid: str):
    wav = config.AUDIO_DIR / f"{guid}.wav"
    if not wav.exists():
        raise HTTPException(404, "no audio")
    return FileResponse(wav, media_type="audio/wav")


def _load_ref(guid: str) -> tuple[np.ndarray, int]:
    wav = config.AUDIO_DIR / f"{guid}.wav"
    if not wav.exists():
        raise HTTPException(404, "no audio")
    a, sr = sf.read(wav, dtype="float32")
    return a, sr


@app.get("/api/segments/{guid}/pitch")
def api_pitch(guid: str, t0: float = 0.0, t1: Optional[float] = None):
    a, sr = _load_ref(guid)
    return pitch.extract_f0(a, sr, t0, t1).to_json()


@app.post("/api/segments/{guid}/compare")
async def api_compare(
    guid: str,
    audio: UploadFile = File(...),
    t0: float = Form(...),
    t1: float = Form(...),
    para: int = Form(0),
):
    """Learner recording (any browser container) vs. reference window [t0, t1]."""
    ref_a, sr = _load_ref(guid)
    raw = await audio.read()
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.bin"
        src.write_bytes(raw)
        rec_id = db.add_recording(guid, para, "")
        out = config.RECORDING_DIR / f"{rec_id}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(sr),
             "-c:a", "pcm_s16le", str(out)],
            check=True,
        )
        with db.connect() as con:
            con.execute("UPDATE recording SET path = ? WHERE id = ?", (str(out), rec_id))
    usr_a, _ = sf.read(out, dtype="float32")
    ref = pitch.extract_f0(ref_a, sr, t0, t1)
    usr = pitch.extract_f0(usr_a, sr)
    result = pitch.compare(ref, usr)
    result["recording_id"] = rec_id
    return JSONResponse(result)


app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="app")
