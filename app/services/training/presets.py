"""XTTS v2 GPT fine-tuning presets, sized for a single 16 GB consumer GPU."""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    preset: str = "balanced"
    epochs: int = 10
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 5e-6
    precision: str = "bf16"            # bf16 | fp16 | fp32
    grad_checkpointing: bool = False
    num_workers: int = 2
    save_step: int = 1000
    eval: bool = True
    max_steps: int = 0                 # 0 = unlimited (used by smoke tests)
    seed: int = 1

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _workers() -> int:
    # Windows DataLoader workers use spawn: each re-imports torch (~2 GB RAM). Keep it modest.
    return 2 if sys.platform == "win32" else 4


PRESETS: dict[str, TrainConfig] = {
    # ~8–10 GB peak: small micro-batches, activations recomputed (gradient checkpointing).
    "low_vram": TrainConfig(preset="low_vram", batch_size=2, grad_accum=16, grad_checkpointing=True,
                            precision="bf16", num_workers=_workers()),
    # ~12–14 GB peak on RTX 4060 Ti 16 GB.
    "balanced": TrainConfig(preset="balanced", batch_size=4, grad_accum=8, grad_checkpointing=False,
                            precision="bf16", num_workers=_workers()),
    "custom": TrainConfig(preset="custom", num_workers=_workers()),
}

PRESET_LABELS = {"low_vram": "Low VRAM", "balanced": "Balanced (RTX 4060 Ti 16 GB)", "custom": "Custom"}


def preset(name: str) -> TrainConfig:
    return TrainConfig.from_dict(PRESETS[name].as_dict())


def resolve_precision(requested: str, device: str) -> str:
    """bf16 needs Ampere+; fall back to fp32 rather than risk fp16 NaNs in the XTTS GPT."""
    if device != "cuda":
        return "fp32"
    if requested == "bf16":
        import torch

        return "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    return requested


def oom_downgrade(cfg: TrainConfig) -> TrainConfig | None:
    """Return a more memory-frugal config with the same effective batch, or None if exhausted."""
    c = TrainConfig.from_dict(cfg.as_dict())
    if not c.grad_checkpointing:
        c.grad_checkpointing = True
        return c
    if c.batch_size > 1:
        c.batch_size //= 2
        c.grad_accum *= 2
        return c
    return None
