"""告警判斷器：替告警平台決定「這筆告警要不要叫人」（ML 課進度存檔 v01）。

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


def main():
    df = load_alerts()
    df["decision"] = platform_rule(df)
    print(f"== {LINE}：平台試用 10 週的告警 ==")
    print(f"{len(df)} 筆告警，{df['occurred_at'].dt.date.nunique()} 天；處理結果：")
    print(df["status"].value_counts().to_string())
    print("\n== 平台規則的決定 × 值班人員最後的處理 ==")
    print(pd.crosstab(df["decision"], df["status"], margins=True, margins_name="合計").to_string())
    auto = df[df["decision"] == "auto_alert"]
    print(f"\n自動叫人 {len(auto)} 次，其中誤報 {(auto['status'] == 'dismissed').sum()} 次"
          f"（{(auto['status'] == 'dismissed').mean():.1%}）")
    missed = df[(df["decision"] == "log_only") & (df["status"] == "resolved")]
    print(f"只記錄、沒叫人的告警裡，事後確認是真的：{len(missed)} 筆")


if __name__ == "__main__":
    main()
