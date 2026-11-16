"""GPU smoke test for the target machine (Windows 11 + RTX GPU).

Steps (each reported PASS/FAIL/SKIP, nothing is claimed unless it actually ran):
  1. detect CUDA GPU
  2. load XTTS v2 base model
  3. short Russian inference, verify output audio
  4. minimal real fine-tuning (N optimizer steps) on a tiny dataset built from the test fixture
  5. resume from the saved checkpoint (proves save + reload), then export and synthesize with it

    python scripts\\gpu_smoke_test.py --accept-cpml [--steps 4] [--allow-cpu]

``--accept-cpml`` records that you accept the XTTS v2 Coqui Public Model License (non-commercial).
Results are also written to data/smoke_test_result.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "short_two_speakers.wav"
TEXTS = ["Сегодня мы проверяем обучение модели.", "Это короткая тестовая фраза.", "Голос звучит чисто и ясно.",
         "Проверка контрольной точки.", "Синтез речи работает локально."]


def build_tiny_dataset(out: Path, n: int = 40) -> Path:
    """Fixture slices (3–6 s) + placeholder transcripts. Only for mechanical training tests."""
    from app.services.dataset.builder import XTTS_SR
    from app.utils.audio import load_mono, save_wav

    x, sr = load_mono(FIXTURE, XTTS_SR)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    rows, man = [], []
    for i in range(n):
        dur = 3.0 + (i % 4)
        start = (i * 2.7) % (len(x) / sr - dur)
        f = f"wavs/{i}.wav"
        save_wav(out / f, x[int(start * sr): int((start + dur) * sr)], sr)
        rows.append(f"{f}|{TEXTS[i % len(TEXTS)]}|smoke")
        man.append({"file": f, "start": start, "end": start + dur, "quality": 1.0, "speaker_sim": 1.0})
    (out / "metadata_train.csv").write_text("audio_file|text|speaker_name\n" + "\n".join(rows[4:]) + "\n", "utf-8")
    (out / "metadata_eval.csv").write_text("audio_file|text|speaker_name\n" + "\n".join(rows[:4]) + "\n", "utf-8")
    (out / "manifest.json").write_text(json.dumps({"clips": man}), "utf-8")
    return out


def run(steps: int, allow_cpu: bool, accept_cpml: bool, data_dir: Path | None = None) -> dict:
    import os

    if data_dir:
        os.environ["LVT_DATA_DIR"] = str(data_dir)
    from app.config.settings import Settings
    from app.database.db import Database, get_or_create_project
    from app.services.dataset.builder import validate_dataset
    from app.utils.jobs import JobContext

    s = Settings.load()
    if accept_cpml:
        s.cpml_accepted = True
    res: dict = {"platform": platform.platform(), "python": platform.python_version(), "steps": {}}

    def step(name, fn):
        t = time.time()
        try:
            out = fn()
            res["steps"][name] = {"status": "PASS", "detail": out, "sec": round(time.time() - t, 1)}
            print(f"[PASS] {name}: {out}")
            return True
        except Exception as e:
            res["steps"][name] = {"status": "FAIL", "detail": f"{type(e).__name__}: {e}",
                                  "trace": traceback.format_exc()[-3000:]}
            print(f"[FAIL] {name}: {e}")
            return False

    import torch

    def detect():
        if not torch.cuda.is_available():
            if allow_cpu:
                return "no CUDA GPU - continuing on CPU because --allow-cpu was given"
            raise RuntimeError("torch.cuda.is_available() is False")
        p = torch.cuda.get_device_properties(0)
        return f"{p.name}, {p.total_memory / 1024**3:.1f} GB, sm_{p.major}{p.minor}, torch {torch.__version__} " \
               f"CUDA {torch.version.cuda}, bf16={torch.cuda.is_bf16_supported()}"

    if not step("1_detect_gpu", detect):
        return res
    if not torch.cuda.is_available():
        s.device = "cpu"

    from app.services.inference.xtts_infer import XttsEngine
    from app.services.training import base_models, manager
    from app.services.training.presets import preset

    eng = XttsEngine(s)

    def load():
        d = base_models.ensure_xtts(s, JobContext())
        eng.load(d)
        return f"XTTS v2 loaded on {eng.device}"

    if not step("2_load_model", load):
        return res

    def infer():
        import numpy as np

        a = eng.synthesize("Привет! Это короткая проверка синтеза речи.", [FIXTURE], "ru", JobContext())
        dur = len(a) / 24000
        rms = float(np.sqrt(np.mean(a ** 2)))
        if not (0.8 < dur < 15) or rms < 0.01 or not np.isfinite(a).all():
            raise RuntimeError(f"implausible output: {dur:.2f}s rms={rms:.4f}")
        if torch.cuda.is_available():
            return f"{dur:.2f}s audio, rms {rms:.3f}, peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB"
        return f"{dur:.2f}s audio, rms {rms:.3f}"

    step("3_inference", infer)
    eng.unload()

    db = Database(s.db_path)
    pid = get_or_create_project(db, "smoke-test")
    tmp = Path(tempfile.mkdtemp(prefix="lvt_smoke_", dir=s.sub("datasets")))
    ds_path = build_tiny_dataset(tmp)
    rep = validate_dataset(ds_path)
    ds_id = db.insert("datasets", project_id=pid, name="smoke", path=str(ds_path), language="ru", n_train=rep.n_train,
                      n_eval=rep.n_eval, total_duration=rep.total_sec, created_at=time.time())
    cfg = preset("balanced" if torch.cuda.is_available() else "low_vram")
    cfg.max_steps, cfg.epochs, cfg.save_step, cfg.num_workers = steps, 100, 10**6, 0
    if not torch.cuda.is_available():
        cfg.batch_size, cfg.grad_accum = 1, 1
    run_id = manager.create_run(db, s, pid, ds_id, cfg)
    events: list[dict] = []

    def train():
        tp = manager.TrainingProcess(db, s, run_id, lambda e: events.append(e) if e["event"] != "log" else None)
        st = tp.execute()
        if st != "stopped":
            raise RuntimeError(f"status {st}: {db.scalar('SELECT error FROM training_runs WHERE id=?', (run_id,))}\n"
                               + "\n".join(list(tp.tail)[-15:]))
        losses = [e["loss"] for e in events if e["event"] == "train_step" and "loss" in e]
        peak = max((e.get("gpu_peak_gb", 0) for e in events if e["event"] == "progress"), default=0)
        ck = manager.best_checkpoint(db, run_id)
        if not ck or not ck.exists():
            raise RuntimeError("no checkpoint written")
        return f"{len(losses)} logged steps, losses {[round(x, 3) for x in losses]}, peak VRAM {peak:.1f} GB, {ck.name}"

    if not step("4_train", train):
        return res

    def resume_and_export():
        import json as _j

        r = db.q1("SELECT * FROM training_runs WHERE id=?", (run_id,))
        c = _j.loads(r["config_json"])
        c["max_steps"] = r["current_step"] + 1
        db.update("training_runs", run_id, config_json=_j.dumps(c))
        evs: list[dict] = []
        tp = manager.TrainingProcess(db, s, run_id, lambda e: evs.append(e) if e["event"] == "started" else None)
        st = tp.execute()
        started = [e for e in evs if e["event"] == "started"]
        if st != "stopped" or not started or not started[0]["resumed"] or started[0]["start_step"] <= 0:
            raise RuntimeError(f"resume failed: status={st} events={started}")
        mid = manager.export_model(db, s, run_id, "smoke voice")
        from app.services.inference.xtts_infer import model_refs

        d, refs, _ = model_refs(db, mid)
        e2 = XttsEngine(s)
        e2.load(d)
        a = e2.synthesize("Проверка дообученной модели.", refs, "ru", JobContext())
        e2.unload()
        return f"resumed at step {started[0]['start_step']}, exported model #{mid}, synthesized {len(a) / 24000:.2f}s"

    step("5_resume_export_infer", resume_and_export)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--accept-cpml", action="store_true")
    a = ap.parse_args()
    if not a.accept_cpml:
        print("XTTS v2 weights are licensed under the Coqui Public Model License (non-commercial). "
              "Re-run with --accept-cpml if you accept it.")
        return 2
    res = run(a.steps, a.allow_cpu, a.accept_cpml)
    from app.config.settings import Settings

    out = Settings.load().data_path / "smoke_test_result.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
    ok = all(v["status"] == "PASS" for v in res["steps"].values()) and len(res["steps"]) == 5
    print(("ALL PASSED" if ok else "NOT ALL STEPS PASSED") + f" - details in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
