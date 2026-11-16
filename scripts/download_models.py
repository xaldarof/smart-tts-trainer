"""Pre-download model weights so the app can run offline afterwards.

    python scripts/download_models.py --processing
    python scripts/download_models.py --xtts --accept-cpml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import Settings  # noqa: E402
from app.services.training import base_models  # noqa: E402
from app.utils.jobs import JobContext  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processing", action="store_true", help="Whisper, Demucs, ECAPA, Silero VAD")
    ap.add_argument("--xtts", action="store_true", help="XTTS v2 base weights (CPML, non-commercial)")
    ap.add_argument("--accept-cpml", action="store_true")
    a = ap.parse_args()
    s = Settings.load()
    ctx = JobContext(lambda f, m: print(f"[{f * 100:5.1f}%] {m}", flush=True))
    if a.processing:
        base_models.prefetch_processing_models(s, ctx)
    if a.xtts:
        if not a.accept_cpml:
            print(base_models.XTTS_LICENSE_TEXT)
            print("Re-run with --accept-cpml to accept the license and download.")
            return 2
        s.cpml_accepted = True
        s.save()
        print("Downloaded to", base_models.ensure_xtts(s, ctx))
    if not (a.processing or a.xtts):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
