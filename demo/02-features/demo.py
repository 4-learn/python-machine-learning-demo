"""02｜挑選特徵：挑出告警發生當下就看得到的欄位，交給模型學。"""
import pandas as pd

df = pd.read_csv("story/data/factory/alerts.csv")
print(f"告警歷史：{df.shape[0]} 筆 × {df.shape[1]} 欄")

print("\n== 1. 挑出特徵 X 和答案 y ==")
NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CATEGORICAL = ["event_type", "location", "source"]
X = df[NUMERIC + CATEGORICAL]
y = df["label"]
print(f"X：{X.shape}，y：{y.shape}")
print(X.head(3).to_string(index=False))
print(f"要叫人的有 {y.sum()} 筆")

print("\n== 2. 事後才知道的欄位：resolved_at ==")
print(pd.crosstab(df["resolved_at"].notna(), df["label"]))
