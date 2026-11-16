"""GPU helpers: device selection, memory stats, cleanup between stages, OOM retry."""
from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from contextlib import contextmanager
from typing import Callable, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")

# One GPU-heavy stage at a time (Demucs, pyannote, Whisper, XTTS inference).
GPU_LOCK = threading.Lock()


def _register_windows_cuda_dlls() -> None:
    """ctranslate2 (faster-whisper) needs cuBLAS/cuDNN DLLs. PyTorch CUDA wheels ship
    them in ``torch/lib``; make them discoverable so users don't need a system CUDA."""
    if sys.platform != "win32":
        return
    try:
        import torch  # noqa: F401

        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:  # pragma: no cover - windows only
        log.warning("Could not register torch CUDA DLL directory: %s", e)


_register_windows_cuda_dlls()


def pick_device(pref: str = "auto") -> str:
    import torch

    if pref == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        raise RuntimeError(
            "CUDA was requested but is not available. Check the NVIDIA driver (`nvidia-smi`) and that the "
            "CUDA build of PyTorch is installed (run scripts\\install.ps1 again), or set Device=Auto/CPU in Settings."
        )
    return "cpu"


def gpu_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        idx = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(idx)
        props = torch.cuda.get_device_properties(idx)
        return {
            "available": True,
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_gb": total / 1024**3,
            "free_gb": free / 1024**3,
            "used_gb": (total - free) / 1024**3,
            "allocated_gb": torch.cuda.memory_allocated(idx) / 1024**3,
            "bf16": torch.cuda.is_bf16_supported(),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in msg or "cuda_error_out_of_memory" in msg or "cublas_status_alloc_failed" in msg


@contextmanager
def gpu_stage(name: str):
    """Serialize GPU-heavy work and release cached memory afterwards."""
    with GPU_LOCK:
        log.info("GPU stage start: %s", name)
        try:
            yield
        finally:
            free_gpu_memory()
            log.info("GPU stage end: %s", name)


def run_with_oom_fallback(fn: Callable[[str], T], device: str, what: str) -> T:
    """Run ``fn(device)``; on CUDA OOM free memory and retry once on CPU."""
    try:
        return fn(device)
    except Exception as e:
        if device == "cuda" and is_oom(e):
            log.warning("%s ran out of GPU memory; retrying on CPU (slower).", what)
            free_gpu_memory()
            return fn("cpu")
        raise
