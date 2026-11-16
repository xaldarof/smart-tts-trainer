"""XTTS v2 GPT fine-tuning entry point (runs in a child process).

Usage::

    python -m app.services.training.xtts_runner spec.json

This follows the official Coqui recipe (``recipes/ljspeech/xtts_v2/train_gpt_xtts.py``)
using ``TTS.tts.layers.xtts.trainer.gpt_trainer.GPTTrainer`` and ``trainer.Trainer``:
the GPT-2 decoder of XTTS is fine-tuned on (text, audio-code) pairs from the dataset;
the DVAE and HiFi-GAN decoder stay frozen. This is real gradient-based fine-tuning,
not reference-audio cloning.

Why a child process: training holds ~10–14 GB of VRAM and long-lived CUDA state.
A separate process isolates crashes, guarantees all GPU memory is returned when it
ends, and lets the GUI stay responsive.

Protocol: machine-readable events are printed to stdout as ``@@LVT {json}`` lines.
Graceful stop: the parent creates ``spec["stop_file"]``; the step callback sees it and
raises KeyboardInterrupt, which makes the Coqui Trainer save a checkpoint and exit 130.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

EVENT_PREFIX = "@@LVT "
EXIT_STOPPED = 130


def emit(event: str, **data) -> None:
    sys.stdout.write(EVENT_PREFIX + json.dumps({"event": event, "t": time.time(), **data}) + "\n")
    sys.stdout.flush()


def _gpu_mem() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {"gpu_alloc_gb": torch.cuda.memory_allocated() / 1024**3,
            "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3}


def enable_gradient_checkpointing(gpt2) -> int:
    """Activation checkpointing on every GPT-2 block of the XTTS decoder.

    ``PreTrainedModel.gradient_checkpointing_enable()`` cannot be used: it calls
    ``enable_input_require_grads()``, which needs ``wte``, and XTTS deletes ``wte`` because it feeds
    its own (trainable) ``inputs_embeds``. We set the per-layer flags directly instead.
    Returns the number of blocks switched to checkpointing.
    """
    import functools

    from torch.utils.checkpoint import checkpoint

    gpt2.config.use_cache = False
    fn = functools.partial(checkpoint, use_reentrant=False)
    n = 0
    for m in gpt2.modules():
        if hasattr(m, "gradient_checkpointing"):
            m._gradient_checkpointing_func = fn
            m.gradient_checkpointing = True
            n += 1
    if n == 0:
        raise RuntimeError("This transformers version exposes no gradient-checkpointing hooks on GPT-2 blocks.")
    return n


def _patch_resume_config(exp_dir: Path, spec: dict) -> None:
    """Trainer reloads config.json on resume; apply memory-related overrides (e.g. after OOM)."""
    p = exp_dir / "config.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["batch_size"] = spec["batch_size"]
    cfg["eval_batch_size"] = max(1, spec["batch_size"] // 2)
    cfg["num_loader_workers"] = spec["num_workers"]
    cfg["epochs"] = spec["epochs"]
    cfg["mixed_precision"] = spec["precision"] != "fp32"
    cfg["precision"] = spec["precision"] if spec["precision"] != "fp32" else "fp16"
    p.write_text(json.dumps(cfg, indent=4), encoding="utf-8")


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    base = Path(spec["base_model_dir"])
    dataset = Path(spec["dataset_path"])
    out_root = Path(spec["output_dir"])
    stop_file = Path(spec["stop_file"])
    stop_file.unlink(missing_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from trainer import Trainer, TrainerArgs
    from trainer.logging.tensorboard_logger import TensorboardLogger

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig

    use_cuda = torch.cuda.is_available() and spec.get("device", "cuda") == "cuda"
    if not use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    precision = spec["precision"] if use_cuda else "fp32"
    emit("init", cuda=use_cuda, device=torch.cuda.get_device_name(0) if use_cuda else "cpu", precision=precision)

    model_args = GPTArgs(
        max_conditioning_length=132300,      # 6 s
        min_conditioning_length=66150,       # 3 s
        debug_loading_failures=False,
        max_wav_length=255995,               # ~11.6 s
        max_text_length=200,
        mel_norm_file=str(base / "mel_stats.pth"),
        dvae_checkpoint=str(base / "dvae.pth"),
        xtts_checkpoint=str(base / "model.pth"),
        tokenizer_file=str(base / "vocab.json"),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    ds_cfg = BaseDatasetConfig(formatter="coqui", dataset_name="lvt", path=str(dataset),
                               meta_file_train="metadata_train.csv", meta_file_val="metadata_eval.csv",
                               language=spec["language"])
    config = GPTTrainerConfig(
        output_path=str(out_root),
        model_args=model_args,
        run_name=spec.get("run_name", "xtts_ft"),
        project_name="local-voice-trainer",
        run_description="XTTS v2 GPT fine-tuning (Local Voice Trainer)",
        dashboard_logger="tensorboard",
        audio=audio_config,
        epochs=spec["epochs"],
        batch_size=spec["batch_size"],
        batch_group_size=48,
        eval_batch_size=max(1, spec["batch_size"] // 2),
        num_loader_workers=spec["num_workers"],
        num_eval_loader_workers=0,
        eval_split_max_size=256,
        print_step=25,
        plot_step=1,
        log_model_step=10**9,
        save_step=spec["save_step"],
        save_n_checkpoints=1,
        save_checkpoints=True,
        save_best_after=0,
        print_eval=False,
        run_eval=spec.get("eval", True),
        test_delay_epochs=10**9,             # no per-epoch test synthesis (saves time & VRAM)
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=spec["lr"],
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
        mixed_precision=precision != "fp32",
        precision=precision if precision != "fp32" else "fp16",
        training_seed=spec.get("seed", 1),
        datasets=[ds_cfg],
    )

    # coqui-tts >= 0.28: load_tts_samples takes the config (with .datasets), not a dataset list
    train_samples, eval_samples = load_tts_samples(config, eval_split=True,
                                                   eval_split_max_size=config.eval_split_max_size)
    emit("data", n_train=len(train_samples), n_eval=len(eval_samples))
    if not train_samples:
        emit("error", message="Training split is empty after loading.")
        return 2

    model = GPTTrainer.init_from_config(config)
    if spec.get("grad_checkpointing"):
        enable_gradient_checkpointing(model.xtts.gpt.gpt)
        emit("info", message="Gradient checkpointing enabled on the XTTS GPT-2 decoder")

    continue_path = spec.get("continue_path") or ""
    if continue_path:
        _patch_resume_config(Path(continue_path), spec)

    class EventLogger(TensorboardLogger):
        def train_step_stats(self, step, stats):
            super().train_step_stats(step, stats)
            emit("train_step", step=int(step), **{k: float(v) for k, v in stats.items()
                                                  if isinstance(v, (int, float)) or hasattr(v, "item")})

        def eval_stats(self, step, stats):
            super().eval_stats(step, stats)
            emit("eval", step=int(step), **{k: float(v) for k, v in stats.items()
                                            if isinstance(v, (int, float)) or hasattr(v, "item")})

    seen_ckpts: set[str] = set()
    t0 = time.time()

    def on_step_end(trainer):
        loader = getattr(trainer, "train_loader", None)
        steps_per_epoch = len(loader) if loader is not None else None
        emit("progress", step=trainer.total_steps_done, epoch=trainer.epochs_done, epochs=trainer.config.epochs,
             steps_per_epoch=steps_per_epoch, grad_accum=trainer.grad_accum_steps, elapsed=time.time() - t0,
             **_gpu_mem())
        for p in Path(trainer.output_path).glob("*.pth"):
            if p.name not in seen_ckpts:
                seen_ckpts.add(p.name)
                emit("checkpoint", path=str(p))
        max_steps = spec.get("max_steps") or 0
        if stop_file.exists() or (max_steps and trainer.total_steps_done >= max_steps):
            emit("stopping", reason="stop requested" if stop_file.exists() else "max_steps reached")
            raise KeyboardInterrupt

    def on_epoch_end(trainer):
        out = Path(trainer.output_path)
        # The trainer writes best_model_<step>.pth AND a full copy best_model.pth (~5.5 GB each).
        # Resume only uses the numbered file, so drop the duplicate to halve disk usage.
        if (out / "best_model.pth").exists() and list(out.glob("best_model_*.pth")):
            (out / "best_model.pth").unlink()
            seen_ckpts.discard("best_model.pth")
        for p in out.glob("*.pth"):
            if p.name not in seen_ckpts:
                seen_ckpts.add(p.name)
                emit("checkpoint", path=str(p))
        emit("epoch_end", epoch=trainer.epochs_done, best_loss=_best(trainer))

    def _best(trainer):
        b = getattr(trainer, "best_loss", None)
        if isinstance(b, dict):
            b = b.get("eval_loss") if b.get("eval_loss") is not None else b.get("train_loss")
        try:
            return float(b) if b is not None and b != float("inf") else None
        except (TypeError, ValueError):
            return None

    run_dir = Path(continue_path) if continue_path else None
    trainer_args = TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False,
                               grad_accum_steps=spec["grad_accum"], continue_path=continue_path)
    tb_dir = (run_dir or out_root) / "tb"
    trainer = Trainer(trainer_args, config, output_path=str(out_root), model=model, train_samples=train_samples,
                      eval_samples=eval_samples, parse_command_line_args=False,
                      dashboard_logger=EventLogger(str(tb_dir), "xtts"),
                      callbacks={"on_train_step_end": on_step_end, "on_epoch_end": on_epoch_end})
    emit("started", output_path=str(trainer.output_path), resumed=bool(continue_path),
         start_step=trainer.total_steps_done, start_epoch=getattr(trainer, "epochs_done", 0))
    trainer.fit()   # exits the process itself on interrupt (130) or error (1)
    for p in Path(trainer.output_path).glob("*.pth"):
        if p.name not in seen_ckpts:
            emit("checkpoint", path=str(p))
    emit("finished", output_path=str(trainer.output_path), step=trainer.total_steps_done)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m app.services.training.xtts_runner spec.json", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
