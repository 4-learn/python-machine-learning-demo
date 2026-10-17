"""06｜KNN 為什麼需要縮放？類別欄怎麼變成數字？用 Pipeline 把前處理和模型綁在一起。"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CATEGORICAL, FEATURES, NUMERIC, TARGET, fit_valid_split, load_events, score_row, time_split

train, _ = time_split(load_events())
fit, valid = fit_valid_split(train)
one = valid[valid.event_id == "E02916"]          # 05 節那筆：高架作業區、信心 0.926，KNN 卻說不通知

print("== 1. 每個欄位的尺度差很多（fit 期間）==")
spread = fit[NUMERIC].agg(["min", "max", "std"]).T.round(3)
print(spread.to_string())

print("\n== 2. 距離被誰決定？E02916 和它最近的鄰居 E01787 ==")
near = fit[fit.event_id == "E01787"]
diff2 = (one[NUMERIC].values[0] - near[NUMERIC].values[0]) ** 2
share = pd.DataFrame({"差距": np.sqrt(diff2).round(3), "占距離²": (diff2 / diff2.sum()).round(3)}, index=NUMERIC)
print(share.to_string())

print("\n== 3. StandardScaler：每欄減平均、除以標準差（只從 fit 學）==")
scaler = StandardScaler().fit(fit[NUMERIC])
learned = pd.DataFrame({"平均": scaler.mean_.round(3), "標準差": scaler.scale_.round(3)}, index=NUMERIC)
print(learned.to_string())
z = pd.DataFrame(scaler.transform(one[NUMERIC]), columns=NUMERIC).round(2)
print("E02916 縮放後：")
print(z.to_string(index=False))

print("\n== 4. 縮放後，E02916 的鄰居換了一批 ==")
knn = KNeighborsClassifier(n_neighbors=15).fit(scaler.transform(fit[NUMERIC]), fit[TARGET])
_, idx = knn.kneighbors(scaler.transform(one[NUMERIC]))
nb = fit.iloc[idx[0]]
print(nb[["event_id", "zone", "confidence", "duration_sec", "tenure_days", TARGET]].head(5).to_string(index=False))
print(f"15 個鄰居裡 {int(nb[TARGET].sum())} 個該通知 → 預測 {int(knn.predict(scaler.transform(one[NUMERIC]))[0])}")

print("\n== 5. one-hot：一個類別欄變成好幾個 0/1 欄 ==")
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(fit[["zone"]])
sample = fit.drop_duplicates("zone").head(3)[["zone"]]
print(pd.DataFrame(enc.transform(sample), columns=enc.categories_[0], index=sample.zone).astype(int).to_string())

print("\n== 6. ColumnTransformer + Pipeline ==")
model = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])),
    ("knn", KNeighborsClassifier(n_neighbors=15)),
])
model.fit(fit[FEATURES], fit[TARGET])
names = model["prep"].get_feature_names_out()
print(f"輸入 {len(FEATURES)} 欄 → 模型看到 {len(names)} 欄")
print(", ".join(names[len(NUMERIC):]))

print("\n== 7. 成績表（valid）：一次改一件事 ==")
steps = {
    "KNN k=15 未縮放": Pipeline([("knn", KNeighborsClassifier(n_neighbors=15))]),
    "KNN 只縮放數值": Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=15))]),
}
rows = []
for name, m in steps.items():
    m.fit(fit[NUMERIC], fit[TARGET])
    rows.append(score_row(name, valid[TARGET], m.predict(valid[NUMERIC])))
rows.append(score_row("KNN k=15", valid[TARGET], model.predict(valid[FEATURES])))
print(pd.DataFrame(rows).to_string(index=False))
