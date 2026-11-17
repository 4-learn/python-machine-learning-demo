"""05｜scikit-learn KNeighborsClassifier：看附近的鄰居投票。"""
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

# 過去的 10 筆告警（手打的小例子）：信心值高、持續久的，事後多半是真的。
past = pd.DataFrame({
    "confidence":       [0.92, 0.88, 0.85, 0.79, 0.45, 0.52, 0.38, 0.61, 0.70, 0.40],
    "duration_seconds": [45,   60,   38,   52,   6,    9,    4,    12,   8,    5],
    "days_on_site":     [120,  900,  1500, 400,  130,  880,  1480, 420,  1010, 1430],
    "label":            [1,    1,    1,    1,    0,    0,    0,    0,    0,    0],
}, index=[f"A{i}" for i in range(1, 11)])
new = pd.DataFrame({"confidence": [0.90, 0.66], "duration_seconds": [50, 31],
                    "days_on_site": [1450, 700]}, index=["新告警 N1", "新告警 N2"])

print("== 1. 過去的告警 ==")
print(past.to_string())

print("\n== 2. fit 再 predict：3 個鄰居投票 ==")
TWO = ["confidence", "duration_seconds"]
knn = KNeighborsClassifier(n_neighbors=3).fit(past[TWO], past["label"])
pred = knn.predict(new[TWO])
print(pd.Series(pred, index=new.index).to_string())

print("\n== 3. kneighbors()：N2 的鄰居是誰 ==")
dist, idx = knn.kneighbors(new[TWO].loc[["新告警 N2"]])
near = past.iloc[idx[0]][TWO + ["label"]].assign(distance=dist[0].round(3))
print(near.to_string())
print(f"鄰居裡要叫人的比例：{near['label'].mean():.3f}")

print("\n== 4. n_neighbors：問幾個鄰居，答案會變 ==")
for k in [1, 3, 5, 7]:
    m = KNeighborsClassifier(n_neighbors=k).fit(past[TWO], past["label"])
    _, idx = m.kneighbors(new[TWO].loc[["新告警 N2"]])
    votes = past["label"].iloc[idx[0]]
    print(f"k={k}：鄰居 {', '.join(votes.index)}，要叫人 {votes.sum()}/{k} → predict {m.predict(new[TWO].loc[['新告警 N2']])[0]}")

print("\n== 5. 加上 days_on_site：距離被大數字綁架 ==")
THREE = TWO + ["days_on_site"]
knn3 = KNeighborsClassifier(n_neighbors=3).fit(past[THREE], past["label"])
dist, idx = knn3.kneighbors(new[THREE].loc[["新告警 N1"]])
near = past.iloc[idx[0]][THREE + ["label"]].assign(distance=dist[0].round(3))
print(near.to_string())
print(f"兩欄時 N1 → {knn.predict(new[TWO])[0]}；加上 days_on_site 後 N1 → {knn3.predict(new[THREE])[0]}")
first = near.index[0]
gap = (new[THREE].loc["新告警 N1"] - past[THREE].loc[first]).abs()
print(f"N1 和 {first} 各欄差多少：", ", ".join(f"{c} {gap[c]:g}" for c in THREE))

print("\n== 6. 真實資料（工廠線，valid）：大數字欄位的影響 ==")
df = pd.read_csv("story/data/factory/alerts.csv", parse_dates=["occurred_at"])
fit = df[df["occurred_at"] < "2025-12-15"]
valid = df[(df["occurred_at"] >= "2025-12-15") & (df["occurred_at"] < "2025-12-29")]
rows = []
for name, cols in [("信心值、持續秒數", TWO),
                   ("＋light_lux", TWO + ["light_lux"]),
                   ("＋days_on_site", TWO + ["days_on_site"]),
                   ("＋兩欄都加", TWO + ["light_lux", "days_on_site"])]:
    pred = KNeighborsClassifier(n_neighbors=15).fit(fit[cols], fit["label"]).predict(valid[cols])
    rows.append({"features": name, "precision": round(precision_score(valid["label"], pred), 3),
                 "recall": round(recall_score(valid["label"], pred), 3), "alerts": int(pred.sum())})
print(pd.DataFrame(rows).to_string(index=False))
print("各欄的範圍：", ", ".join(f"{c} {fit[c].min():g}～{fit[c].max():g}" for c in TWO + ["light_lux", "days_on_site"]))
