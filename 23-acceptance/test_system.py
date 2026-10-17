"""23｜系統的驗收測試（Mock 偵測，不需要 YOLO）。

    python -m pytest 23-acceptance -q
"""
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from system import CONFIG, DetectionPipeline, ThresholdCalibrator, load_frames  # noqa: E402


@pytest.fixture(scope="module")
def system():
    return DetectionPipeline(mock=True)


def test_rerun_gives_same_decisions(system):
    """同一批照片，重新建立系統再跑一次，逐欄相同。"""
    assert system.process_frames(load_frames()).equals(DetectionPipeline(mock=True).process_frames(load_frames()))


def test_threshold_is_the_approved_one(system):
    assert system.calibrator.value == 0.2
    assert system.calibrator.provenance["approved_by"] == "陳主任"


def test_unapproved_proposal_is_refused():
    """22 節的提案（status: proposed、approved_by: null）不能直接拿來上線。"""
    with pytest.raises(PermissionError):
        ThresholdCalibrator(HERE.parent / "22-drift" / "notify-proposed.yaml")


def test_rest_area_never_notifies(system):
    d = system.process_frames(load_frames())
    assert not ((d.zone == "休息區") & (d.action == "notify")).any()


def test_test_set_opened_with_record():
    """test 每打開一次都要先有紀錄，而且紀錄要寫明打開後沒有改東西。"""
    log = yaml.safe_load((HERE / "test-log.yaml").read_text(encoding="utf-8"))["openings"]
    assert all(o["changed_after"] == "無" for o in log)
    assert yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))["threshold"]["approved_at"] < log[-1]["date"]
