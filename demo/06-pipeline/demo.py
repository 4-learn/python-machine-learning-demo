"""06｜scikit-learn Pipeline：縮放、one-hot，接成一條。"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 05 節的 10 筆告警，多了一個類別欄 location。
past = pd.DataFrame({
    "confidence":       [0.92, 0.88, 0.85, 0.79, 0.45, 0.52, 0.38, 0.61, 0.70, 0.40],
    "duration_seconds": [45,   60,   38,   52,   6,    9,    4,    12,   8,    5],
    "days_on_site":     [120,  900,  1500, 400,  130,  880,  1480, 420,  1010, 1430],
    "location":         ["aisle_01", "dock_01", "aisle_02", "aisle_01", "rest_01",
                         "rest_01", "aisle_02", "dock_01", "rest_01", "aisle_01"],
    "label":            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
}, index=[f"A{i}" for i in range(1, 11)])
new = pd.DataFrame({"confidence": [0.90], "duration_seconds": [50], "days_on_site": [1450],
                    "location": ["aisle_01"]}, index=["新告警 N1"])

print("== 1. StandardScaler：換成「離平均幾個標準差」 ==")
NUM = ["confidence", "duration_seconds", "days_on_site"]
scaler = StandardScaler().fit(past[NUM])
z_past = scaler.transform(past[NUM])
z_new = scaler.transform(new[NUM])
print(pd.DataFrame({"mean_": scaler.mean_, "scale_": scaler.scale_}, index=NUM).to_string(float_format="%.3f"))
print("新告警 N1 →", z_new.round(3)[0])
for name, X_past, X_new in [("未縮放", past[NUM], new[NUM]),
                            ("縮放後", z_past, z_new)]:
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_past, past["label"])
    _, idx = knn.kneighbors(X_new)
    print(f"{name}：鄰居 {', '.join(past.index[idx[0]])} → predict {knn.predict(X_new)[0]}")

print("\n== 2. OneHotEncoder：一個類別變成一欄 0／1 ==")
enc = OneHotEncoder(handle_unknown="ignore").fit(past[["location"]])
test_loc = pd.DataFrame({"location": ["dock_01", "aisle_03"]})
onehot = enc.transform(test_loc).toarray()
print("學到的類別：", list(enc.categories_[0]))
print(pd.DataFrame(onehot, columns=enc.get_feature_names_out(),
                   index=test_loc["location"]).astype(int).to_string())

print("\n== 3. ColumnTransformer＋make_pipeline：前處理和模型接成一條 ==")
pre = ColumnTransformer([("num", StandardScaler(), NUM),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), ["location"])])
pipe = make_pipeline(pre, KNeighborsClassifier(n_neighbors=3)).fit(past, past["label"])
pred = pipe.predict(new)
names = pre.get_feature_names_out()
print(f"輸入 {past[NUM + ['location']].shape[1]} 欄 → 輸出 {len(names)} 欄")
print(", ".join(names))
print("N1 →", pred[0])
