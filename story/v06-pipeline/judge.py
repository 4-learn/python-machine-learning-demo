"""告警判斷器：替 API 課的平台決定「這筆告警要不要叫人」（ML 課進度存檔 v06）。

執行（在這個資料夾裡）：
    LINE=factory python judge.py     # 工廠線
    LINE=care python judge.py        # 健康線
"""
import json
import math
import os
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LINE = os.getenv("LINE", "factory")
HERE = Path(__file__).resolve().parent
STORY = HERE.parent
PROFILE = json.loads((STORY / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
DATA = STORY / "data" / LINE / "alerts.csv"


def load_alerts():
    """讀入平台 10 週的告警歷史。"""
    df = pd.read_csv(DATA)
    df["occurred_at"] = pd.to_datetime(df["occurred_at"])
    return df


def platform_rule(df):
    """API 課 v0.12 的規則：信心值 0.8 以上自動開告警、0.5 到 0.8 待人確認、其餘只記錄。"""
    return pd.cut(df["confidence"], bins=[0, 0.5, 0.8, 1.01], right=False,
                  labels=["log_only", "require_review", "auto_alert"])


# 02：哪些欄位可以當特徵。事後才知道的處理紀錄不行。
NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CATEGORICAL = ["event_type", "location", "source"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "label"
LEAKY = ["acked_at", "resolved_at", "status"]


def xy(df):
    return df[FEATURES], df[TARGET]


# 03：用時間切。前 6 週學（fit）、第 7–8 週比較模型（valid）、最後 2 週是考卷（test）。
VALID_START = pd.Timestamp("2025-12-15")
TEST_START = pd.Timestamp("2025-12-29")


def split_by_time(df):
    fit = df[df["occurred_at"] < VALID_START]
    valid = df[(df["occurred_at"] >= VALID_START) & (df["occurred_at"] < TEST_START)]
    test = df[df["occurred_at"] >= TEST_START]
    return fit, valid, test


def score_row(name, y_true, y_pred):
    """成績表的一列：叫人的次數，以及叫得準不準、漏了多少。"""
    return {"model": name,
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred), 3),
            "alerts": int(pd.Series(y_pred).sum())}


def baseline_rows(fit, valid):
    X_fit, y_fit = xy(fit)
    X_valid, y_valid = xy(valid)
    dummy = DummyClassifier().fit(X_fit, y_fit)
    rule = (platform_rule(valid) == "auto_alert").astype(int)
    return [score_row("多數類", y_valid, dummy.predict(X_valid)),
            score_row("平台規則 conf>=0.8", y_valid, rule)]


def make_preprocessor(scale=True):
    """數值欄縮放（樹模型可以不縮放），類別欄 one-hot。"""
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def model_rows(fit, valid):
    X_fit, y_fit = xy(fit)
    X_valid, y_valid = xy(valid)
    rows = []
    knn_raw = KNeighborsClassifier(n_neighbors=15).fit(fit[NUMERIC], y_fit)
    rows.append(score_row("KNN 未縮放", y_valid, knn_raw.predict(valid[NUMERIC])))
    knn = make_pipeline(make_preprocessor(), KNeighborsClassifier(n_neighbors=15)).fit(X_fit, y_fit)
    rows.append(score_row("KNN", y_valid, knn.predict(X_valid)))
    return rows


def main():
    df = load_alerts()
    fit, valid, _ = split_by_time(df)
    print(f"== {LINE}：成績表（valid）==")
    print(pd.DataFrame(baseline_rows(fit, valid) + model_rows(fit, valid)).to_string(index=False))


if __name__ == "__main__":
    main()
