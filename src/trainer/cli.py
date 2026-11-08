"""Command line entry point.

  trainer ingest <segment-url> [--no-translate] [--force]
  trainer ingest <YYYY-MM-DD>  [--no-translate] [--force] [--include-full]
  trainer translate [guid ...] [--force]
  trainer list   [--from YYYY-MM-DD] [--to YYYY-MM-DD]
  trainer serve  [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from . import config, db


def _parse_date(s: str) -> date | None:
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def cmd_ingest(args: argparse.Namespace) -> int:
    from . import pipeline

    d = _parse_date(args.target)
    if d:
        guids = pipeline.process_day(d, translate=not args.no_translate, force=args.force,
                                     include_full=args.include_full)
        print(f"{len(guids)} segments processed for {d}")
        return 0
    guid = pipeline.process_url(args.target, translate=not args.no_translate, force=args.force)
    print(guid)
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    from . import pipeline

    db.init_db()
    guids = args.guids or [s.guid for s in db.list_segments() if s.status in ("enriched", "ready")]
    n = 0
    for g in guids:
        try:
            if pipeline.translate_existing(g, force=args.force):
                n += 1
        except Exception as e:
            print(f"FAILED {g}: {e}", file=sys.stderr)
    print(f"{n} segments translated")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    db.init_db()
    segs = db.list_segments(_parse_date(args.date_from) if args.date_from else None,
                            _parse_date(args.date_to) if args.date_to else None)
    for s in segs:
        flags = json.loads(s.qc_flags) if s.qc_flags else []
        mark = "!" if flags else " "
        print(f"{s.broadcast_date} {s.guid} {s.status:<10}{mark} {s.duration_s or 0:6.0f}s  {s.headline}")
        for f in flags:
            print(f"      QC: {f}")
        if s.error:
            print(f"      {s.status}: {s.error}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("trainer.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="ingest one segment URL or one broadcast date")
    pi.add_argument("target")
    pi.add_argument("--no-translate", action="store_true")
    pi.add_argument("--force", action="store_true", help="re-run even if already ready")
    pi.add_argument("--include-full", action="store_true", help="also ingest the 30-min 完整版")
    pi.set_defaults(fn=cmd_ingest)

    pt = sub.add_parser("translate", help="add English to processed segments (all untranslated if no guids)")
    pt.add_argument("guids", nargs="*")
    pt.add_argument("--force", action="store_true")
    pt.set_defaults(fn=cmd_translate)

    pl = sub.add_parser("list")
    pl.add_argument("--from", dest="date_from")
    pl.add_argument("--to", dest="date_to")
    pl.set_defaults(fn=cmd_list)

    ps = sub.add_parser("serve")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8765)
    ps.add_argument("--reload", action="store_true")
    ps.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    config.ensure_dirs()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
