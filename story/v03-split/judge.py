"""告警判斷器：替 API 課的平台決定「這筆告警要不要叫人」（ML 課進度存檔 v03）。

執行（在這個資料夾裡）：
    LINE=factory python judge.py     # 工廠線
    LINE=care python judge.py        # 健康線
"""
import json
import math
import os
from pathlib import Path

import pandas as pd

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


def main():
    df = load_alerts()
    fit, valid, test = split_by_time(df)
    print(f"== {LINE}：用時間切出三份 ==")
    for name, part in [("fit", fit), ("valid", valid), ("test", test)]:
        print(f"{name:<6}{len(part):>5} 筆  {part['occurred_at'].min():%m-%d}～{part['occurred_at'].max():%m-%d}"
              f"  要叫人 {part[TARGET].mean():.3f}")
    shared = set(fit["incident_id"]) & (set(valid["incident_id"]) | set(test["incident_id"]))
    print(f"同一件事被切到兩邊：{len(shared)} 件")


if __name__ == "__main__":
    main()
