"""01｜pandas 的 cut 與 crosstab：把分數分成三段，再和真正的結果交叉比對。"""
import pandas as pd

alerts = pd.DataFrame({
    "alert_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "confidence": [0.95, 0.83, 0.81, 0.72, 0.66, 0.55, 0.49, 0.41, 0.35, 0.92],
    "status": ["resolved", "dismissed", "resolved", "resolved", "dismissed",
               "dismissed", "resolved", "dismissed", "dismissed", "dismissed"],
})

print("== 1. pd.cut：依信心值切成三段 ==")
alerts["decision"] = pd.cut(alerts["confidence"], bins=[0, 0.5, 0.8, 1.01], right=False,
                            labels=["log_only", "require_review", "auto_alert"])
print(alerts.to_string(index=False))

print("\n== 2. value_counts：每一段有幾筆 ==")
print(alerts["decision"].value_counts(sort=False).to_string())

print("\n== 3. pd.crosstab：決定 × 結果 ==")
print(pd.crosstab(alerts["decision"], alerts["status"]).to_string())

print("\n== 4. margins=True：加上合計 ==")
print(pd.crosstab(alerts["decision"], alerts["status"], margins=True, margins_name="合計"))

print("\n== 5. 換成平台真的資料：10 週、3540 筆 ==")
df = pd.read_csv("story/data/factory/alerts.csv")
df["decision"] = pd.cut(df["confidence"], bins=[0, 0.5, 0.8, 1.01], right=False,
                        labels=["log_only", "require_review", "auto_alert"])
print(pd.crosstab(df["decision"], df["status"], margins=True, margins_name="合計"))
