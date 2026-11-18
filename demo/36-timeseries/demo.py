"""36｜scikit-learn TimeSeriesSplit：時間序列只能用過去考未來；殘差告訴你哪些時段老是猜錯。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

print("== 1. TimeSeriesSplit(n_splits=3)：12 個小時怎麼切 ==")
for k, (tr, va) in enumerate(TimeSeriesSplit(n_splits=3).split(np.arange(12)), 1):
    print(f"第 {k} 折：學 {tr.tolist()}  考 {va.tolist()}")
tr, va = next(KFold(n_splits=3, shuffle=True, random_state=0).split(np.arange(12)))
print(f"對照 KFold(shuffle=True) 第 1 折：學 {tr.tolist()}  考 {va.tolist()}")

s = pd.read_csv("story/data/factory/stream/detections.csv")
s["ts"] = pd.to_datetime(s["ts"])
h = s.drop_duplicates().set_index("ts")["confidence"].resample("1h").count().rename("count").to_frame()
h["t"] = np.arange(len(h))             # 第幾個小時（時間的編號）
h["hour"] = h.index.hour
h["last_hour"] = h["count"].shift(1)
h = h.dropna()

print("\n== 2. 隨機切 vs 照時間切：同一個模型，MAE 差多少 ==")
knn = KNeighborsRegressor(n_neighbors=2)
for name, cv in [("KFold(shuffle=True)", KFold(n_splits=5, shuffle=True, random_state=0)),
                 ("TimeSeriesSplit", TimeSeriesSplit(n_splits=5))]:
    mae = -cross_val_score(knn, h[["t"]], h["count"], cv=cv, scoring="neg_mean_absolute_error")
    print(f"{name:<20}MAE {mae.mean():.3f}")

print("\n== 3. 殘差（實際 − 預測）依小時平均：正的＝猜太少、負的＝猜太多 ==")
split = int(len(h) * 0.7)
train, test = h.iloc[:split], h.iloc[split:]
model = LinearRegression().fit(train[["last_hour"]], train["count"])
resid = test["count"] - model.predict(test[["last_hour"]])
by_hour = resid.groupby(test["hour"]).mean().round(2)
print(by_hour.loc[[7, 8, 9, 17, 18, 19]].to_string())
print(f"全部 {len(test)} 小時的殘差平均 {resid.mean():.3f}；每個小時只有 {test['hour'].value_counts().max()} 筆")
