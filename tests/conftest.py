import os
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")                      # GUI tests run headless
os.environ["LIVE_TRANSLATOR_HOME"] = tempfile.mkdtemp(prefix="lt_test_")   # never touch the real config
os.environ["HF_HUB_CACHE"] = tempfile.mkdtemp(prefix="lt_hf_")            # never touch the real model cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_server import FakeOpenAI  # noqa: E402


@pytest.fixture
def server():
    s = FakeOpenAI().start()
    yield s
    s.stop()


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
