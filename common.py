"""全課共用：讀資料、特徵清單、切割方式。

每一節的 demo 都從 repo 根目錄執行，例如：
    python 05-knn/knn_first.py

這裡只放「全課要一致」的東西：資料怎麼讀、哪些欄位是特徵、測試集怎麼切。
模型與評估留在各節自己的程式裡，讓你看得到每一步。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# 02 節決定的特徵。事後才知道的欄位（ticket_id、notified_at、handled_minutes）不在這裡。
NUMERIC = ["confidence", "duration_sec", "light_lux", "people_nearby",
           "tenure_days", "prior_violations_30d", "hour"]
CATEGORICAL = ["zone", "shift", "site"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "needs_notify"
LEAKY = ["ticket_id", "notified_at", "handled_minutes"]

# 03 節決定的切割：前 8 週訓練、最後 2 週測試（用過去預測未來，同一事件不會跨兩邊）。
TEST_START = pd.Timestamp("2025-12-29 00:00:00+08:00")


def load_events(path: Path | str = DATA / "events.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticket_id": "str", "notified_at": "str"})
    df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S%z")
    return df


def time_split(df: pd.DataFrame):
    """回傳 (train, test) 兩個 DataFrame。"""
    is_test = df["event_time"] >= TEST_START
    return df[~is_test].copy(), df[is_test].copy()


def xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]


# 04 節起：模型比較用「驗證集」＝訓練期的最後 2 週。測試集留到 10 節只算一次。
VALID_START = pd.Timestamp("2025-12-15 00:00:00+08:00")


def fit_valid_split(train: pd.DataFrame):
    """把 time_split() 的 train 再切成 (fit, valid)。"""
    is_valid = train["event_time"] >= VALID_START
    return train[~is_valid].copy(), train[is_valid].copy()


def current_rule(df: pd.DataFrame) -> pd.Series:
    """01 節：公司現行規則——偵測信心 ≥ 0.7 就通知。（0.7 從哪裡來？見 12 節）"""
    return (df["confidence"] >= 0.7).astype(int)


def make_preprocessor(scale: bool = True):
    """06 節介紹：數值欄縮放、類別欄 one-hot。只在 Pipeline 裡使用。"""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def score_row(name: str, y_true, y_pred) -> dict:
    """成績表的一列（04 節起全課共用）。"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    return {"model": name,
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred), 3),
            "alerts": int(pd.Series(y_pred).sum())}
