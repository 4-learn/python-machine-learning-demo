"""35｜scikit-learn LinearRegression：預測下一小時的偵測數（第一次迴歸：答案是數字，不是類別）。"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

print("== 1. 兩條線：每小時偵測數，依一天裡的小時平均 ==")
rows = {}
for line in ["factory", "care"]:
    s = pd.read_csv(f"story/data/{line}/stream/detections.csv")
    s["ts"] = pd.to_datetime(s["ts"])
    per_hour = s.drop_duplicates().set_index("ts")["confidence"].resample("1h").count()
    rows[line] = per_hour.groupby(per_hour.index.hour).mean().round(1)
print(pd.DataFrame(rows).loc[[2, 7, 8, 14, 18, 20, 21, 22]].to_string())

print("\n== 2. shift：把「上一小時」搬到同一列 ==")
toy = pd.DataFrame({"count": [12, 15, 50, 55, 52, 18]},
                   index=pd.date_range("2026-01-05 06:00", periods=6, freq="1h"))
toy["last_hour"] = toy["count"].shift(1)
toy["next_hour"] = toy["count"].shift(-1)
print(toy.to_string())

print("\n== 3. LinearRegression：學到一條「乘法加法」的規則 ==")
s = pd.read_csv("story/data/factory/stream/detections.csv")
s["ts"] = pd.to_datetime(s["ts"])
h = s.drop_duplicates().set_index("ts")["confidence"].resample("1h").count().rename("count").to_frame()
h["last_hour"] = h["count"].shift(1)
h["same_hour_yesterday"] = h["count"].shift(24)
h = h.dropna()
train, test = h[h.index < "2026-01-09"], h[h.index >= "2026-01-09"]
model = LinearRegression().fit(train[["last_hour"]], train["count"])
print(f"每小時一列：學 {len(train)} 小時（01-06～01-08）、考 {len(test)} 小時（01-09 起）")
print(f"coef_ = {model.coef_.round(3)}、intercept_ = {model.intercept_:.3f}")
x = 50
print(f"上一小時 {x} 筆 → predict {model.predict(pd.DataFrame({'last_hour': [x]}))[0]:.3f}"
      f"；手算 {model.coef_[0]:.3f} × {x} + {model.intercept_:.3f} = {model.coef_[0] * x + model.intercept_:.3f}")

print("\n== 4. mean_absolute_error：和兩條 baseline 比（test，越小越好）==")
cols = ["last_hour", "same_hour_yesterday"]
model2 = LinearRegression().fit(train[cols], train["count"])
rows = [("一律猜訓練期的平均", [train["count"].mean()] * len(test)),
        ("照抄上一小時", test["last_hour"]),
        ("LinearRegression：上一小時", model.predict(test[["last_hour"]])),
        ("LinearRegression：上一小時＋昨天同一小時", model2.predict(test[cols]))]
for name, pred in rows:
    print(f"MAE {mean_absolute_error(test['count'], pred):6.3f}  {name}")
print("第二個模型的係數：", ", ".join(f"{c} {w:+.3f}" for c, w in zip(cols, model2.coef_)))
