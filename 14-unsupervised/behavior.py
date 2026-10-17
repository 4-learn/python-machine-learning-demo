"""14｜KMeans 看行為型態：把 80 位人員依「事件的樣子」分群。沒有標籤，只能描述，不能宣稱對錯。"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from common import TARGET, load_events, time_split

train, _ = time_split(load_events())
people = train.groupby("worker_id").agg(
    事件數=("event_id", "size"),
    平均持續=("duration_sec", "mean"),
    平均信心=("confidence", "mean"),
    夜班比例=("shift", lambda s: (s == "夜班").mean()),
    休息區比例=("zone", lambda s: (s == "休息區").mean()),
    高架比例=("zone", lambda s: (s == "高架作業區").mean()),
    年資=("tenure_days", "first"),
)
COLS = list(people.columns)

print(f"== 1. 一列＝一位人員（前 8 週，{len(people)} 人），{len(COLS)} 個行為欄位 ==")
print(people.head(3).round(2).to_string())

print("\n== 2. 沒縮放就分群：被「年資」一個欄位主導 ==")
raw = KMeans(n_clusters=2, n_init=10, random_state=0).fit(people[COLS])
print(people.assign(群=raw.labels_).groupby("群")[["年資", "平均持續", "平均信心"]].mean().round(1).to_string())
print(f"silhouette {silhouette_score(people[COLS], raw.labels_):.3f}（分數高，只是因為年資的數字最大）")

print("\n== 3. 標準化後，k=2～5 ==")
X = StandardScaler().fit_transform(people[COLS])
for k in range(2, 6):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f"k={k}  silhouette {silhouette_score(X, km.labels_):.3f}  各群人數 {sorted(np.bincount(km.labels_).tolist(), reverse=True)}")

print("\n== 4. k=2 的兩群長什麼樣子 ==")
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
people["群"] = km.labels_
profile = people.groupby("群")[COLS].mean()
profile.insert(0, "人數", people.群.value_counts().sort_index())
print(profile.round(2).to_string())

print("\n== 5. 換 random_state，k=4 的分群還一樣嗎？（1.00＝完全相同）==")
base = KMeans(n_clusters=4, n_init=1, random_state=0).fit(X).labels_
same = [adjusted_rand_score(base, KMeans(n_clusters=4, n_init=1, random_state=s).fit(X).labels_) for s in range(1, 5)]
print("random_state 1～4 與 0 的一致度：", ", ".join(f"{a:.2f}" for a in same))

print("\n== 6. 事後才拿標籤來看（分群時沒用到）==")
rate = train.groupby("worker_id")[TARGET].mean()
after = pd.DataFrame({"人數": people.群.value_counts().sort_index(),
                      "該通知比例（平均）": people.groupby("群").apply(lambda g: rate[g.index].mean()).round(3)})
print(after.to_string())
