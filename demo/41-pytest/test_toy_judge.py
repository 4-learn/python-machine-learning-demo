"""41 Demo：判斷器的自動驗收。不測準不準，測規則。"""
import joblib
import pytest

from toy_judge import ToyJudge, sha256_12, train


@pytest.fixture(scope="module")
def judge():
    """訓練一次，這個檔案裡的測試共用。"""
    return ToyJudge(train())


def test_decision_is_one_of_three(judge):
    for conf in [0.1, 0.5, 0.9]:
        assert judge.decide(conf, 30.0) in {"auto_alert", "require_review", "log_only"}


def test_same_input_same_output(judge):
    assert judge.score(0.7, 30.0) == judge.score(0.7, 30.0)


def test_higher_confidence_never_lowers_score(judge):
    assert judge.score(0.99, 30.0) >= judge.score(0.31, 30.0)


def test_rest_area_is_not_auto_alert(judge):
    assert judge.decide(0.95, 8.0, rest_area=1) != "auto_alert"


def test_confidence_out_of_range(judge):
    with pytest.raises(ValueError, match="0 到 1"):
        judge.score(1.7, 30.0)


def test_tampered_model_is_rejected(judge, tmp_path):
    path = tmp_path / "toy.joblib"
    joblib.dump(judge.model, path)
    expected = sha256_12(path)
    with open(path, "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError):
        ToyJudge.from_file(path, expected)
