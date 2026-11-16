"""First-run / on-demand system diagnostics. Every check returns an actionable message."""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.utils.audio import CREATE_NO_WINDOW


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    severity: str = "error"   # error | warning | info

    def as_row(self) -> tuple[str, str, str]:
        status = "OK" if self.ok else ("WARN" if self.severity == "warning" else "FAIL")
        return self.name, status, self.detail


MIN_FREE_DISK_GB = 30
RECOMMENDED_VRAM_GB = 12


def check_python() -> Check:
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) <= (3, 13)
    return Check("Python", ok, f"{platform.python_version()} ({'supported' if ok else 'use Python 3.12'})")


def check_ffmpeg() -> Check:
    exe = shutil.which("ffmpeg")
    if not exe:
        return Check("FFmpeg", False, "Not found on PATH. Install: winget install Gyan.FFmpeg, then restart.")
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10,
                             creationflags=CREATE_NO_WINDOW).stdout.splitlines()[0]
    except Exception as e:
        return Check("FFmpeg", False, f"Found at {exe} but failed to run: {e}")
    return Check("FFmpeg", True, out)


def check_nvidia_driver() -> Check:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return Check("NVIDIA driver", False,
                     "nvidia-smi not found. Install the latest NVIDIA Game Ready/Studio driver (>= 570 for CUDA 12.8).",
                     "warning")
    try:
        out = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15, creationflags=CREATE_NO_WINDOW).stdout.strip()
        return Check("NVIDIA driver", bool(out), out or "nvidia-smi returned no GPUs", "warning")
    except Exception as e:
        return Check("NVIDIA driver", False, f"nvidia-smi failed: {e}", "warning")


def check_torch_cuda() -> list[Check]:
    try:
        import torch
    except Exception as e:
        return [Check("PyTorch", False, f"Import failed: {e}. Re-run scripts\\install.ps1.")]
    checks = [Check("PyTorch", True, f"{torch.__version__} (CUDA build: {torch.version.cuda or 'none'})")]
    if not torch.cuda.is_available():
        checks.append(Check("CUDA", False,
                            "CUDA not available: processing works on CPU (slow), training is not practical. "
                            "Install the NVIDIA driver and the CUDA PyTorch wheel (install.ps1 does this).", "warning"))
        return checks
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024**3
    checks.append(Check("CUDA GPU", True, f"{props.name}, {vram:.1f} GB VRAM, compute {props.major}.{props.minor}"))
    if (props.major, props.minor) < (7, 0):
        checks.append(Check("GPU arch", False, "GPU older than Volta (sm_70) is not supported by this PyTorch build."))
    if vram < 8:
        checks.append(Check("VRAM", False, f"{vram:.1f} GB is below the 8 GB minimum for XTTS fine-tuning."))
    elif vram < RECOMMENDED_VRAM_GB:
        checks.append(Check("VRAM", False, f"{vram:.1f} GB: use the 'Low VRAM' training preset.", "warning"))
    try:
        x = torch.randn(256, 256, device="cuda")
        (x @ x).sum().item()
        checks.append(Check("CUDA compute", True, "Test matmul succeeded"))
    except Exception as e:
        checks.append(Check("CUDA compute", False, f"Kernel launch failed: {e}. Driver may be too old for this CUDA build."))
    return checks


def check_packages() -> list[Check]:
    pkgs = {
        "yt_dlp": "yt-dlp", "faster_whisper": "faster-whisper", "silero_vad": "silero-vad", "TTS": "coqui-tts",
        "demucs": "demucs", "speechbrain": "speechbrain", "pyannote.audio": "pyannote.audio", "PySide6": "PySide6",
    }
    out = []
    for mod, name in pkgs.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "") or getattr(getattr(m, "version", None), "__version__", "")
            out.append(Check(name, True, str(ver) or "installed"))
        except Exception as e:
            out.append(Check(name, False, f"Import failed: {e}. Re-run scripts\\install.ps1."))
    return out


def check_disk(path: Path) -> Check:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / 1024**3
    ok = free >= MIN_FREE_DISK_GB
    return Check("Disk space", ok, f"{free:.1f} GB free at {path}" + ("" if ok else
                 f" — at least {MIN_FREE_DISK_GB} GB recommended (model weights ~6 GB, training needs ~17 GB for checkpoints while a run is active)."),
                 "warning")


def ensure_free_space(path: Path, needed_gb: float, what: str) -> None:
    free = shutil.disk_usage(path).free / 1024**3
    if free < needed_gb:
        raise OSError(f"Not enough disk space for {what}: {free:.1f} GB free at {path}, ~{needed_gb:.1f} GB needed. "
                      "Free up space or move the data directory in Settings.")


def run_all(data_dir: Path, include_cuda: bool = True) -> list[Check]:
    checks = [check_python(), check_ffmpeg(), check_nvidia_driver()]
    if include_cuda:
        checks += check_torch_cuda()
    checks += check_packages()
    checks.append(check_disk(data_dir))
    return checks


if __name__ == "__main__":
    from app.config.settings import Settings

    for c in run_all(Settings.load().data_path):
        print("{:<16} {:<5} {}".format(*c.as_row()))
