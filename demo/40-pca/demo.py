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
print(f"原本 {pts.shape} → transform 後 {z.shape}")
print(pd.DataFrame(np.hstack([pts.to_numpy()[:3], z[:3]]), columns=["x", "y", "pc1", "pc2"]).round(3).to_string(index=False))

alerts = pd.read_csv("story/data/factory/alerts.csv")
fit = alerts[alerts["occurred_at"] < "2025-12-15"]
COLS = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]

print("\n== 3. 6 個數值欄，沒縮放就 PCA ==")
print("各欄標準差：")
print(fit[COLS].std().round(3).to_string())
raw = PCA(n_components=3).fit(fit[COLS])
print("explained_variance_ratio_：", raw.explained_variance_ratio_.round(3).tolist())

print("\n== 4. 先 StandardScaler 再 PCA ==")
scaler = StandardScaler().fit(fit[COLS])
scaled = PCA(n_components=6).fit(scaler.transform(fit[COLS]))
print("explained_variance_ratio_：", scaled.explained_variance_ratio_.round(3).tolist())
print("累加：", scaled.explained_variance_ratio_.cumsum().round(3).tolist())
print("\npc1 的方向（每一欄占多少）：")
print(pd.DataFrame({"沒縮放": raw.components_[0], "先縮放": scaled.components_[0]}, index=COLS).round(3).to_string())

print("\n== 5. PCA 沒看標籤：fit 只給 X，事後再拿 label 對照 ==")
z = PCA(n_components=2).fit_transform(scaler.transform(fit[COLS]))
out = pd.DataFrame(z, columns=["pc1", "pc2"]).assign(label=fit["label"].to_numpy())
print(out.groupby("label")[["pc1", "pc2"]].mean().round(3).to_string())
