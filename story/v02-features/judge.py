"""告警判斷器：替告警平台決定「這筆告警要不要叫人」（ML 課進度存檔 v02）。

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
    """平台預設的規則：信心值 0.8 以上自動開告警、0.5 到 0.8 待人確認、其餘只記錄。"""
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


def main():
    df = load_alerts()
    X, y = xy(df)
    print(f"== {LINE}：告警歷史變成 X 和 y ==")
    print(f"X：{X.shape[0]} 列 × {X.shape[1]} 欄；y：要叫人的比例 {y.mean():.3f}")
    print("數值特徵：", ", ".join(NUMERIC))
    print("類別特徵：", ", ".join(CATEGORICAL))
    print("不能當特徵（事後才知道）：", ", ".join(LEAKY))
    print("\n== 為什麼 resolved_at 不能用 ==")
    print(pd.crosstab(df["resolved_at"].notna().map({True: "有結案時間", False: "沒有"}), y).to_string())


if __name__ == "__main__":
    main()
