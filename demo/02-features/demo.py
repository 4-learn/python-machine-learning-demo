"""02｜pandas：把告警歷史整理成 X（特徵）和 y（標籤），並用 crosstab 找出「事後才知道」的欄位。"""
import pandas as pd

df = pd.read_csv("story/data/factory/alerts.csv")

print("== 1. 一列＝一筆告警 ==")
print(f"{df.shape[0]} 列 × {df.shape[1]} 欄")
print(df[["alert_id", "incident_id", "event_type", "confidence", "resolved_at", "status", "label"]]
      .head(4).to_string(index=False))

print("\n== 2. select_dtypes：數值欄和文字欄 ==")
print("數值：", df.select_dtypes(include="number").columns.tolist())
print("文字：", df.select_dtypes(exclude="number").columns.tolist())

print("\n== 3. df[cols]：只挑要的欄位（白名單）==")
NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CATEGORICAL = ["event_type", "location", "source"]
X = df[NUMERIC + CATEGORICAL]
y = df["label"]
print(f"X：{X.shape}，y：{y.shape}")
print(X.head(3).to_string(index=False))
print(f"y 的值：{y.value_counts().to_dict()}，要叫人的比例 {y.mean():.3f}")

print("\n== 4. drop(columns=)：拿掉不要的欄位（黑名單）==")
rest = df.drop(columns=["label", "acked_at", "resolved_at", "status"])
print("剩下：", rest.columns.tolist())
print("比白名單多出：", [c for c in rest.columns if c not in X.columns])

print("\n== 5. crosstab：resolved_at 有值，label 就是 1 ==")
print(pd.crosstab(df["resolved_at"].notna(), df["label"]))

print("\n== 6. crosstab：acked_at 有值，label 是 1 的機會大很多 ==")
print(pd.crosstab(df["acked_at"].notna(), df["label"], normalize="index").round(3))
