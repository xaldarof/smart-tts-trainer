from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs downloaded model weights (set LVT_INTEGRATION=1)")
    config.addinivalue_line("markers", "gpu: needs an NVIDIA CUDA GPU")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("LVT_INTEGRATION") != "1":
        skip = pytest.mark.skip(reason="set LVT_INTEGRATION=1 to run integration tests (downloads/uses real models)")
        for it in items:
            if "integration" in it.keywords:
                it.add_marker(skip)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LVT_DATA_DIR", str(tmp_path / "data"))
    from app.config.settings import Settings

    return Settings.load()


@pytest.fixture
def db(settings):
    from app.database.db import Database

    d = Database(settings.db_path)
    yield d
    d.close()


@pytest.fixture
def project(db):
    from app.database.db import get_or_create_project

    return get_or_create_project(db, "test")


def tone(sec: float, sr: int = 22050, freq: float = 220.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
