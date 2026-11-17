"""36｜scikit-learn TimeSeriesSplit：時間序列只能用過去考未來；殘差告訴你哪些時段老是猜錯。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

print("== 1. TimeSeriesSplit(n_splits=3)：12 個小時怎麼切 ==")
for k, (tr, va) in enumerate(TimeSeriesSplit(n_splits=3).split(np.arange(12)), 1):
    print(f"第 {k} 折：學 {tr.tolist()}  考 {va.tolist()}")

print("\n== 2. 對照：KFold(shuffle=True) 會拿未來來學 ==")
tr, va = next(KFold(n_splits=3, shuffle=True, random_state=0).split(np.arange(12)))
print(f"第 1 折：學 {tr.tolist()}  考 {va.tolist()}")

s = pd.read_csv("story/data/factory/stream/detections.csv")
s["ts"] = pd.to_datetime(s["ts"])
h = s.drop_duplicates().set_index("ts")["confidence"].resample("1h").count().rename("count").to_frame()
h["t"] = np.arange(len(h))             # 第幾個小時（時間的編號）
h["hour"] = h.index.hour
h["last_hour"] = h["count"].shift(1)
h = h.dropna()

print("\n== 3. 隨機切 vs 照時間切：同一個模型，MAE 差多少 ==")
knn = KNeighborsRegressor(n_neighbors=2)
for name, cv in [("KFold(shuffle=True)", KFold(n_splits=5, shuffle=True, random_state=0)),
                 ("TimeSeriesSplit", TimeSeriesSplit(n_splits=5))]:
    mae = -cross_val_score(knn, h[["t"]], h["count"], cv=cv, scoring="neg_mean_absolute_error")
    print(f"{name:<20}MAE {mae.mean():.3f}")

print("\n== 4. 殘差＝實際 − 預測（只用上一小時，後 30% 當考卷）==")
split = int(len(h) * 0.7)
train, test = h.iloc[:split], h.iloc[split:]
model = LinearRegression().fit(train[["last_hour"]], train["count"])
resid = test["count"] - model.predict(test[["last_hour"]])
print(pd.DataFrame({"count": test["count"], "pred": model.predict(test[["last_hour"]]).round(1),
                    "resid": resid.round(1)}).loc["2026-01-10 06:00":"2026-01-10 09:00"].to_string())

print("\n== 5. 殘差依小時平均：正的＝猜太少、負的＝猜太多 ==")
by_hour = resid.groupby(test["hour"]).mean().round(2)
print(by_hour.loc[[7, 8, 9, 17, 18, 19]].to_string())
print(f"全部 {len(test)} 小時的殘差平均 {resid.mean():.3f}；每個小時只有 {test['hour'].value_counts().max()} 筆")

print("\n== 6. 最後一格：串流結束在那一小時的開頭 ==")
for line in ["factory", "care"]:
    raw = pd.read_csv(f"story/data/{line}/stream/detections.csv")
    raw["ts"] = pd.to_datetime(raw["ts"])
    last = raw.drop_duplicates().set_index("ts")["confidence"].resample("1h").count()
    print(f"{line:<8}串流最後一筆 {raw['ts'].max():%m-%d %H:%M:%S}；最後兩格 "
          + "、".join(f"{ts:%m-%d %H:%M} {n} 筆" for ts, n in last.tail(2).items()))
