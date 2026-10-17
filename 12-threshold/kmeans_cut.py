"""12｜反例：舊版用 KMeans 在 confidence 上找切點。它找到的是「資料的形狀」，不是「對錯的分界」。"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

from common import TARGET, fit_valid_split, load_events, time_split

train, _ = time_split(load_events())
fit, valid = fit_valid_split(train)


def stats(df, t):
    y, p = df[TARGET], (df.confidence >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"通知": int(tp + fp), "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0, "成本": int(10 * fn + fp)}


print("== 1. KMeans(2) 在 valid 的 confidence 上分兩群 ==")
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(valid[["confidence"]])   # 注意：只給 X，沒有 y
centers = np.sort(km.cluster_centers_.ravel())
cut = round(float(centers.mean()), 3)
print(f"群中心 {centers[0]:.3f} 與 {centers[1]:.3f} → 切點（中點）{cut}")
valid["群"] = np.where(valid.confidence >= cut, "高信心群", "低信心群")
print(valid.groupby("群").agg(事件=("event_id", "size"), 該通知比例=(TARGET, "mean")).round(3).to_string())

print("\n== 2. 同樣只看 confidence，用標籤＋成本挑的門檻 ==")
rows = [{"門檻": t, **stats(valid, t)} for t in np.round(np.arange(0.30, 0.96, 0.05), 2)]
best = min(rows, key=lambda r: r["成本"])
print(pd.DataFrame([{"做法": "KMeans 切點", "門檻": cut, **stats(valid, cut)},
                    {"做法": "現行規則", "門檻": 0.70, **stats(valid, 0.70)},
                    {"做法": "標籤＋成本 10:1", **best}]).to_string(index=False))

print("\n== 3. 同一個切點，在不同區域的意思完全不同 ==")
out = []
for zone in ["物料區", "高架作業區", "休息區"]:
    part = valid[valid.zone == zone]
    out.append({"區域": zone, "事件": len(part), "confidence 中位數": round(part.confidence.median(), 3),
                **{k: v for k, v in stats(part, cut).items() if k != "成本"}})
print(pd.DataFrame(out).to_string(index=False))

print("\n== 4. 標籤全部改掉，KMeans 的切點也不會動 ==")
flipped = valid.assign(**{TARGET: 1 - valid[TARGET]})
km2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(flipped[["confidence"]])
print(f"標籤反轉後切點 {np.sort(km2.cluster_centers_.ravel()).mean():.3f}（原本 {cut}）")
