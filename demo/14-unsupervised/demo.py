"""14｜scikit-learn 的 KMeans 與 IsolationForest：沒有標籤的時候，能看出什麼、不能宣稱什麼。"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

alerts = pd.read_csv("story/data/factory/alerts.csv")
alerts = alerts[alerts["occurred_at"] < "2025-12-15"]          # 只用前 6 週（fit）
COLS = ["confidence", "duration_seconds", "light_lux"]
X = alerts[COLS]
km_raw = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
Z = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z)


def describe(groups):
    """每一群的平均值；label 是事後才知道的，分群時沒有用到，只拿來對照。"""
    out = alerts.assign(群=groups).groupby("群").agg(
        筆數=("label", "size"), confidence=("confidence", "mean"),
        duration=("duration_seconds", "mean"), light_lux=("light_lux", "mean"), 事後是真的=("label", "mean"))
    return out.round(3).to_string()


print("== 1. 沒縮放就分群：三欄的數字大小差很多 ==")
print(X.std().round(3).to_string())
print(describe(km_raw.labels_))

print("\n== 2. StandardScaler 之後再分群 ==")
print(describe(km.labels_))

print("\n== 3. IsolationForest：挑出最罕見的告警 ==")
forest = IsolationForest(contamination=0.02, random_state=0).fit(Z)
shown = alerts.assign(score=forest.score_samples(Z).round(3), predict=forest.predict(Z))
print(shown.nsmallest(5, "score")[["alert_id", "event_type", "location"] + COLS + ["score", "predict"]]
      .to_string(index=False))

print("\n== 4. 事後對照：罕見和要叫人是兩回事 ==")
print(pd.crosstab(shown["predict"], shown["label"], margins=True, margins_name="合計").to_string())
