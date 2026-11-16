"""Training run lifecycle: create, launch child process, monitor, stop, resume, export."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

from app.config.settings import PROJECT_ROOT, Settings
from app.database.db import Database
from app.services.dataset.builder import validate_dataset
from app.services.training import base_models
from app.services.training.presets import TrainConfig, oom_downgrade, resolve_precision
from app.services.training.xtts_runner import EVENT_PREFIX, EXIT_STOPPED
from app.utils.audio import CREATE_NO_WINDOW
from app.utils.gpu import pick_device
from app.utils.system_check import ensure_free_space

log = logging.getLogger(__name__)

CKPT_DISK_GB = 6.5   # one full checkpoint incl. AdamW state


class TrainingError(RuntimeError):
    pass


def create_run(db: Database, settings: Settings, project_id: int, dataset_id: int, cfg: TrainConfig) -> int:
    ds = db.q1("SELECT * FROM datasets WHERE id=? AND project_id=?", (dataset_id, project_id))
    if ds is None:
        raise TrainingError("Dataset not found for this project.")
    rep = validate_dataset(Path(ds["path"]))
    if not rep.ok:
        raise TrainingError("Dataset is not valid for training:\n- " + "\n- ".join(rep.errors[:10]))
    run_id = db.insert("training_runs", project_id=project_id, dataset_id=dataset_id, backend="xtts_v2",
                       preset=cfg.preset, config_json=json.dumps(cfg.as_dict()), status="pending",
                       output_dir="", created_at=time.time())
    out = settings.sub("checkpoints") / str(project_id) / f"run_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    db.update("training_runs", run_id, output_dir=str(out))
    return run_id


class TrainingProcess:
    """Runs one training attempt in a child process and mirrors its events into SQLite."""

    def __init__(self, db: Database, settings: Settings, run_id: int,
                 on_event: Callable[[dict], None] | None = None, auto_oom_retry: bool = True):
        self.db, self.settings, self.run_id = db, settings, run_id
        self.on_event = on_event or (lambda e: None)
        self.auto_oom_retry = auto_oom_retry
        self.proc: subprocess.Popen | None = None
        self._stop_requested = False
        self.tail: deque[str] = deque(maxlen=200)

    # ----------------------------------------------------------- control
    @property
    def run(self):
        return self.db.q1("SELECT * FROM training_runs WHERE id=?", (self.run_id,))

    def stop_file(self) -> Path:
        return Path(self.run["output_dir"]) / "STOP"

    def request_stop(self) -> None:
        """Graceful stop: the trainer saves a checkpoint at the next step, then exits."""
        self._stop_requested = True
        self.stop_file().touch()
        try:
            self.db.transition("training_runs", self.run_id, ["running"], "stopping")
        except Exception:
            pass

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    # --------------------------------------------------------------- run
    def execute(self) -> str:
        """Blocking. Returns final status: completed | stopped | failed."""
        while True:
            status, oom = self._attempt()
            if status != "failed" or not oom or not self.auto_oom_retry or self._stop_requested:
                return status
            cfg = TrainConfig.from_dict(json.loads(self.run["config_json"]))
            nxt = oom_downgrade(cfg)
            if nxt is None:
                self._fail("CUDA out of memory even with batch size 1 and gradient checkpointing. Close other GPU "
                           "applications (browsers, games) or shorten max clip length.")
                return "failed"
            log.warning("Training OOM; retrying with batch=%d accum=%d checkpointing=%s", nxt.batch_size,
                        nxt.grad_accum, nxt.grad_checkpointing)
            self.on_event({"event": "info", "message": f"Out of GPU memory - retrying with batch size "
                           f"{nxt.batch_size}, gradient accumulation {nxt.grad_accum}, gradient checkpointing on."})
            self.db.update("training_runs", self.run_id, config_json=json.dumps(nxt.as_dict()), status="interrupted")

    def _spec(self) -> Path:
        r = self.run
        cfg = TrainConfig.from_dict(json.loads(r["config_json"]))
        ds = self.db.q1("SELECT * FROM datasets WHERE id=?", (r["dataset_id"],))
        device = pick_device(self.settings.device)
        exp = r["exp_dir"] if r["exp_dir"] and _has_checkpoint(Path(r["exp_dir"])) else ""
        spec = {
            "base_model_dir": str(base_models.xtts_dir(self.settings)), "dataset_path": ds["path"],
            "output_dir": r["output_dir"], "language": ds["language"], "run_name": f"run_{self.run_id}",
            "stop_file": str(self.stop_file()), "device": device,
            "precision": resolve_precision(cfg.precision, device), "continue_path": exp,
            **{k: v for k, v in cfg.as_dict().items() if k != "precision"},
        }
        p = Path(r["output_dir"]) / "spec.json"
        p.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return p

    def _attempt(self) -> tuple[str, bool]:
        if not base_models.xtts_present(self.settings):
            raise TrainingError("XTTS v2 base weights are missing. Download them in Settings → Models.")
        # latest checkpoint + best checkpoint + one being written
        ensure_free_space(self.settings.data_path, CKPT_DISK_GB * 3, "training checkpoints")
        if pick_device(self.settings.device) != "cuda":
            log.warning("Training on CPU: expect it to be extremely slow; intended only for smoke tests.")
        spec = self._spec()
        self.stop_file().unlink(missing_ok=True)
        self.db.transition("training_runs", self.run_id, ["pending", "interrupted", "failed", "stopped"], "running",
                           started_at=time.time(), error=None, attempts=(self.run["attempts"] or 0) + 1)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log_path = Path(self.run["output_dir"]) / f"train_{int(time.time())}.log"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "app.services.training.xtts_runner", str(spec)], cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            env=env, creationflags=CREATE_NO_WINDOW, bufsize=1,
        )
        oom = False
        with open(log_path, "w", encoding="utf-8") as lf:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                lf.write(line)
                lf.flush()
                line = line.rstrip("\n")
                if line.startswith(EVENT_PREFIX):
                    try:
                        self._handle(json.loads(line[len(EVENT_PREFIX):]))
                    except Exception as e:
                        log.warning("bad training event: %s (%s)", line[:200], e)
                    continue
                self.tail.append(line)
                if "out of memory" in line.lower():
                    oom = True
                self.on_event({"event": "log", "line": line})
        code = self.proc.wait()
        self._register_checkpoints()
        if code == 0:
            self.db.update("training_runs", self.run_id, status="completed", finished_at=time.time())
            return "completed", False
        if code == EXIT_STOPPED or self._stop_requested:
            self.db.update("training_runs", self.run_id, status="stopped", finished_at=time.time())
            return "stopped", False
        tail = "\n".join(list(self.tail)[-25:])
        msg = _explain_failure(tail, code)
        self._fail(msg, tail)
        return "failed", oom

    def _fail(self, msg: str, detail: str = "") -> None:
        self.db.update("training_runs", self.run_id, status="failed", error=msg, finished_at=time.time())
        self.db.log_error(f"training:{self.run_id}", msg, detail)
        self.on_event({"event": "failed", "message": msg})

    def _handle(self, ev: dict) -> None:
        kind = ev.get("event")
        now = time.time()
        if kind == "started":
            self.db.update("training_runs", self.run_id, exp_dir=ev["output_path"])
        elif kind == "progress":
            self.db.update("training_runs", self.run_id, current_step=ev["step"], current_epoch=ev["epoch"])
        elif kind == "train_step":
            vals = {k: v for k, v in ev.items() if k.startswith("loss")}
            if "loss" in vals:
                self.db.update("training_runs", self.run_id, last_loss=vals["loss"])
            with self.db.tx() as c:
                c.executemany("INSERT INTO metrics(run_id, step, epoch, split, name, value, ts) VALUES(?,?,?,?,?,?,?)",
                              [(self.run_id, ev["step"], None, "train", k, v, now) for k, v in vals.items()])
        elif kind == "eval":
            vals = {k: v for k, v in ev.items() if k.startswith("avg_loss") or k.startswith("loss")}
            loss = vals.get("avg_loss", vals.get("loss"))
            r = self.run
            if loss is not None and (r["best_eval_loss"] is None or loss < r["best_eval_loss"]):
                self.db.update("training_runs", self.run_id, best_eval_loss=loss)
            with self.db.tx() as c:
                c.executemany("INSERT INTO metrics(run_id, step, epoch, split, name, value, ts) VALUES(?,?,?,?,?,?,?)",
                              [(self.run_id, ev["step"], None, "eval", k, v, now) for k, v in vals.items()])
        elif kind == "checkpoint":
            self._add_ckpt(Path(ev["path"]))
        self.on_event(ev)

    def _add_ckpt(self, p: Path) -> None:
        m = re.search(r"_(\d+)\.pth$", p.name)
        with self.db.tx() as c:
            c.execute("INSERT OR IGNORE INTO checkpoints(run_id, path, step, created_at) VALUES(?,?,?,?)",
                      (self.run_id, str(p), int(m.group(1)) if m else None, time.time()))

    def _register_checkpoints(self) -> None:
        exp = self.run["exp_dir"]
        if not exp or not Path(exp).exists():
            return
        for p in Path(exp).glob("*.pth"):
            self._add_ckpt(p)
        # drop rows for files the trainer rotated away
        for r in self.db.q("SELECT id, path FROM checkpoints WHERE run_id=?", (self.run_id,)):
            if not Path(r["path"]).exists():
                self.db.execute("DELETE FROM checkpoints WHERE id=?", (r["id"],))


def _has_checkpoint(d: Path) -> bool:
    return d.exists() and any(re.search(r"(checkpoint|best_model)_\d+\.pth$", p.name) for p in d.glob("*.pth"))


def _explain_failure(tail: str, code: int) -> str:
    low = tail.lower()
    if "out of memory" in low:
        return "CUDA out of memory. Use the Low VRAM preset or close other GPU applications."
    if "no space left" in low:
        return "Disk full while writing checkpoints. Free disk space and resume the run."
    if "size mismatch" in low or "unexpected key" in low or "missing key" in low:
        return "Checkpoint is incompatible with the current XTTS model version. Start a new run."
    if "nan" in low and "loss" in low:
        return "Training diverged (NaN loss). Lower the learning rate or switch precision to fp32."
    if "filenotfounderror" in low or "no such file" in low:
        return "A dataset or model file is missing. Re-export the dataset / re-download XTTS weights."
    last = [l for l in tail.splitlines() if l.strip()]
    return f"Training process exited with code {code}: {last[-1] if last else 'no output'}"


# ------------------------------------------------------------------- export
def best_checkpoint(db: Database, run_id: int) -> Path | None:
    r = db.q1("SELECT exp_dir FROM training_runs WHERE id=?", (run_id,))
    if not r or not r["exp_dir"]:
        return None
    d = Path(r["exp_dir"])
    best = sorted(d.glob("best_model*.pth"), key=lambda p: p.stat().st_mtime)
    if best:
        return best[-1]
    ck = sorted(d.glob("checkpoint_*.pth"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    return ck[-1] if ck else None


def export_model(db: Database, settings: Settings, run_id: int, name: str, checkpoint: Path | None = None,
                 n_refs: int = 4) -> int:
    """Strip optimizer state from a checkpoint and package an inference-ready model directory."""
    import torch

    run = db.q1("SELECT * FROM training_runs WHERE id=?", (run_id,))
    if run is None:
        raise TrainingError("Run not found")
    ckpt = checkpoint or best_checkpoint(db, run_id)
    if ckpt is None or not ckpt.exists():
        raise TrainingError("This run has no checkpoint to export yet.")
    ensure_free_space(settings.data_path, 2.5, "model export")
    safe = re.sub(r"[^\w\-]+", "_", name).strip("_") or f"run_{run_id}"
    out = settings.sub("trained_models") / str(run["project_id"]) / f"{safe}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    drop = ("torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae")
    slim = {k: v for k, v in sd.items() if k.split(".")[0] not in drop and
            not (k.startswith("xtts.") and k.split(".")[1] in drop)}
    torch.save({"model": slim}, out / "model.pth")
    base = base_models.xtts_dir(settings)
    for f in ("config.json", "vocab.json"):
        shutil.copyfile(base / f, out / f)
    # Reference clips (needed by XTTS for speaker conditioning at inference time)
    ds = db.q1("SELECT * FROM datasets WHERE id=?", (run["dataset_id"],))
    refs = _pick_reference_clips(Path(ds["path"]), n_refs)
    (out / "refs").mkdir()
    ref_names = []
    for i, src in enumerate(refs):
        dst = out / "refs" / f"ref_{i}.wav"
        shutil.copyfile(src, dst)
        ref_names.append(f"refs/{dst.name}")
    meta = {"backend": "xtts_v2", "run_id": run_id, "checkpoint": str(ckpt), "step": run["current_step"],
            "epoch": run["current_epoch"], "best_eval_loss": run["best_eval_loss"], "language": ds["language"],
            "dataset": ds["path"], "dataset_minutes": ds["total_duration"] / 60, "refs": ref_names,
            "config": json.loads(run["config_json"]), "license": "CPML (non-commercial) - derived from XTTS v2"}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return db.insert("models", project_id=run["project_id"], run_id=run_id, name=name, backend="xtts_v2",
                     kind="finetuned", path=str(out), language=ds["language"], created_at=time.time(),
                     meta_json=json.dumps(meta, ensure_ascii=False))


def _pick_reference_clips(dataset: Path, n: int) -> list[Path]:
    """Prefer the highest-quality 5–11 s clips from the dataset manifest."""
    man = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    clips = [c for c in man["clips"] if 5.0 <= c["end"] - c["start"] <= 11.5] or man["clips"]
    clips.sort(key=lambda c: (c.get("quality") or 0) + (c.get("speaker_sim") or 0), reverse=True)
    return [dataset / c["file"] for c in clips[:n]]


def register_pretrained(db: Database, settings: Settings) -> int:
    row = db.q1("SELECT id FROM models WHERE kind='pretrained' AND backend='xtts_v2'")
    if row:
        return int(row["id"])
    return db.insert("models", name="XTTS v2 (pretrained, no fine-tuning)", backend="xtts_v2", kind="pretrained",
                     path=str(base_models.xtts_dir(settings)), created_at=time.time(),
                     meta_json=json.dumps({"license": "CPML (non-commercial)", "revision": base_models.XTTS_REVISION}))


def run_in_thread(tp: TrainingProcess) -> threading.Thread:
    th = threading.Thread(target=tp.execute, name=f"train-{tp.run_id}", daemon=True)
    th.start()
    return th
