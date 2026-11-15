"""ML 課進度存檔 v41：判斷器的自動驗收（pytest）。

執行（在這個資料夾裡）：
    LINE=factory python -m pytest test_judge.py -q
    LINE=care python -m pytest test_judge.py -q
"""
import json

import pytest

import judge


@pytest.fixture(scope="module")
def judge_ready():
    """先訓練並存檔，再讀回來：所有測試共用同一個判斷器。"""
    judge.train_and_save()
    return judge.Judge()


@pytest.fixture(scope="module")
def events():
    path = judge.STORY / "data" / judge.LINE / "new_events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_decision_is_one_of_three(judge_ready, events):
    for event in events:
        assert judge_ready.decide(event)["decision"] in {"auto_alert", "require_review", "log_only"}


def test_same_input_same_output(judge_ready, events):
    assert judge_ready.decide(events[0]) == judge_ready.decide(events[0])


def test_thresholds_are_ordered(judge_ready):
    assert 0 < judge_ready.config["low"] < judge_ready.config["high"] < 1


def test_higher_confidence_never_lowers_score(judge_ready, events):
    event = dict(events[0])
    low = judge_ready.decide({**event, "confidence": 0.31})["score"]
    high = judge_ready.decide({**event, "confidence": 0.99})["score"]
    assert high >= low


def test_safe_location_is_not_auto_alert(judge_ready, events):
    safe = judge.PROFILE["locations"][-1]           # 設定檔最後一個地點：休息區／交誼廳
    event = {**events[0], "location": safe, "confidence": 0.95,
             "detail": {**events[0]["detail"], "duration_seconds": 8.0}}
    assert judge_ready.decide(event)["decision"] != "auto_alert"


def test_tampered_model_is_rejected(judge_ready, tmp_path):
    import shutil
    for name in ["judge.joblib", "judge.json"]:
        shutil.copy(judge.MODEL_DIR / name, tmp_path / name)
    with open(tmp_path / "judge.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError):
        judge.Judge(model_dir=tmp_path)
