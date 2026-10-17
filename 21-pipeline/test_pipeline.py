"""21｜管線的驗收測試（Mock 模式，不需要 YOLO）。

    python -m pytest 21-pipeline -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import NotifyPipeline, load_mock_events  # noqa: E402


@pytest.fixture(scope="module")
def pipe():
    return NotifyPipeline()


def test_same_input_same_output(pipe):
    """同一批 Event JSON，跑兩次（甚至重新載入管線）決定要一模一樣。"""
    first = pipe.run(load_mock_events())
    second = NotifyPipeline().run(load_mock_events())
    assert first[["event_id", "proba", "action"]].equals(second[["event_id", "proba", "action"]])


def test_every_decision_carries_versions(pipe):
    d = pipe.run(load_mock_events())
    for col in ["model.version", "model.sha256_12", "threshold.value", "rules.version"]:
        assert d[col].notna().all(), col
