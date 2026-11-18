"""40｜scikit-learn 的 PCA：把很多欄壓成幾個主成分（不看標籤）。"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("== 1. 2 欄的小例子：點沿著一條斜線分布 ==")
rng = np.random.default_rng(0)
t = rng.normal(0, 3, 200)
pts = pd.DataFrame({"x": t + rng.normal(0, 0.5, 200), "y": 0.5 * t + rng.normal(0, 0.5, 200)})
pca = PCA(n_components=2).fit(pts)
print("components_（每個主成分的方向）：")
print(pd.DataFrame(pca.components_, columns=["x", "y"], index=["pc1", "pc2"]).round(3).to_string())
print("explained_variance_ratio_：", pca.explained_variance_ratio_.round(3).tolist())

print("\n== 2. transform：每個點換成新座標 ==")
z = pca.transform(pts)
print(pd.DataFrame(np.hstack([pts.to_numpy()[:3], z[:3]]), columns=["x", "y", "pc1", "pc2"]).round(3).to_string(index=False))

alerts = pd.read_csv("story/data/factory/alerts.csv")
fit = alerts[alerts["occurred_at"] < "2025-12-15"]
COLS = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]

print("\n== 3. 告警的 6 個數值欄：沒縮放 vs 先縮放 ==")
raw = PCA(n_components=6).fit(fit[COLS])
scaled_X = StandardScaler().fit_transform(fit[COLS])
scaled = PCA(n_components=6).fit(scaled_X)
print(f"pc1 留下幾成：沒縮放 {raw.explained_variance_ratio_[0]:.3f}、先縮放 {scaled.explained_variance_ratio_[0]:.3f}")
print(pd.DataFrame({"標準差": fit[COLS].std(), "沒縮放的pc1": raw.components_[0],
                    "先縮放的pc1": scaled.components_[0]}, index=COLS).round(3).to_string())

print("\n== 4. PCA 沒看標籤：fit 只給 X，事後再拿 label 對照 ==")
z = PCA(n_components=2).fit_transform(scaled_X)
out = pd.DataFrame(z, columns=["pc1", "pc2"]).assign(label=fit["label"].to_numpy())
print(out.groupby("label")[["pc1", "pc2"]].mean().round(3).to_string())
