"""開場｜故事與資料：平台 10 週的告警，平台規則準不準，哪些欄位可以給模型看。"""
import pandas as pd

df = pd.read_csv("story/data/factory/alerts.csv")

print("== 1. 平台 10 週的告警 ==")
print(f"{len(df)} 筆告警，{df.shape[1]} 個欄位")
print(df[["alert_id", "event_type", "location", "confidence", "status"]].head(5).to_string(index=False))

print("\n== 2. 平台規則的成績 ==")
df["decision"] = pd.cut(df["confidence"], bins=[0, 0.5, 0.8, 1.01], right=False,
                        labels=["log_only", "require_review", "auto_alert"])
print(pd.crosstab(df["decision"], df["status"], margins=True, margins_name="合計"))

print("\n== 3. 17 個欄位分成三堆 ==")
FEATURES = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour",
            "event_type", "location", "source"]
LEAKY = ["acked_at", "resolved_at", "status"]
print("特徵（給模型看）：", ", ".join(FEATURES))
print("答案：label")
print("事後才知道（不能給）：", ", ".join(LEAKY))
