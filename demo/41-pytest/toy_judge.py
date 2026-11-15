"""41 Demo：被測的程式，一個很小的判斷器（只看信心值、持續秒數、是不是休息區）。"""
import hashlib
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

COLS = ["confidence", "duration_seconds", "rest_area"]
TRAIN = pd.DataFrame({
    "confidence":       [0.95, 0.91, 0.88, 0.83, 0.72, 0.66, 0.93, 0.90, 0.55, 0.41, 0.35, 0.86],
    "duration_seconds": [80.0, 65.0, 50.0, 70.0, 45.0, 9.0, 60.0, 75.0, 11.0, 8.0, 6.0, 12.0],
    "rest_area":        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    "label":            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
})
HIGH, LOW = 0.8, 0.3


def train():
    return LogisticRegression(C=10).fit(TRAIN[COLS], TRAIN["label"])


def sha256_12(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


class ToyJudge:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_file(cls, path, expected_sha):
        if sha256_12(path) != expected_sha:
            raise RuntimeError("模型檔和設定檔對不上，拒絕啟動")
        return cls(joblib.load(path))

    def score(self, confidence, duration_seconds, rest_area=0):
        if not 0 <= confidence <= 1:
            raise ValueError(f"信心值要在 0 到 1 之間：{confidence}")
        row = pd.DataFrame([[confidence, duration_seconds, rest_area]], columns=COLS)
        return round(float(self.model.predict_proba(row)[0, 1]), 3)

    def decide(self, confidence, duration_seconds, rest_area=0):
        if os.getenv("TOY_BROKEN"):   # 示範用：有人把判斷器「簡化」成只看信心值
            return "auto_alert" if confidence >= 0.8 else "require_review" if confidence >= 0.5 else "log_only"
        s = self.score(confidence, duration_seconds, rest_area)
        return "auto_alert" if s >= HIGH else "require_review" if s >= LOW else "log_only"
