"""37｜scikit-learn learning_curve：用越來越多的資料訓練，看 valid 分數還會不會往上爬。"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CATEGORICAL = ["event_type", "location", "source"]

df = pd.read_csv("story/data/factory/alerts.csv")
df["occurred_at"] = pd.to_datetime(df["occurred_at"])
train = df[df["occurred_at"] < "2025-12-29"]          # 前 8 週；考卷不碰
X, y, groups = train[NUMERIC + CATEGORICAL], train["label"], train["incident_id"]
pre = ColumnTransformer([("num", "passthrough", NUMERIC),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
model = make_pipeline(pre, HistGradientBoostingClassifier(random_state=0))
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

print("== 1. learning_curve 回傳三樣東西 ==")
sizes, tr_scores, va_scores = learning_curve(
    model, X, y, groups=groups, cv=cv, train_sizes=[0.05, 0.1, 0.25, 0.5, 1.0],
    scoring="average_precision")
print("sizes：", sizes.tolist())
print("train_scores.shape：", tr_scores.shape, "（5 種大小 × 5 折）")

print("\n== 2. 判斷器的告警資料（前 8 週，梯度提升，AP）==")
table = pd.DataFrame({"訓練筆數": sizes, "train": tr_scores.mean(1), "valid": va_scores.mean(1),
                      "valid 標準差": va_scores.std(1), "差距": tr_scores.mean(1) - va_scores.mean(1)})
print(table.round(3).to_string(index=False))

print("\n== 3. 標準差：同一個大小，5 折各考幾分 ==")
for n, row in zip(sizes, va_scores):
    print(f"{n:>5} 筆：", " ".join(f"{v:.3f}" for v in row))

print("\n== 4. 資料太少時，梯度提升連一刀都切不下去（每片葉子至少 20 筆）==")
for n in [39, 40, 113]:
    part = train.sample(n=n, random_state=0)
    m = make_pipeline(pre, HistGradientBoostingClassifier(random_state=0)).fit(part[NUMERIC + CATEGORICAL], part["label"])
    proba = m.predict_proba(part[NUMERIC + CATEGORICAL])[:, 1]
    print(f"隨機 {n:>3} 筆（要叫人的 {int(part['label'].sum()):>2} 筆）：模型給的機率有 {len(np.unique(proba.round(6))):>2} 種不同的值")
