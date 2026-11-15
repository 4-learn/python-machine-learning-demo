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

print("\n== 3. mean_absolute_error：平均差幾筆 ==")
truth, guess = [50, 55, 52], [48, 60, 52]
print(f"實際 {truth}、預測 {guess} → MAE {mean_absolute_error(truth, guess):.3f}")

print("\n== 4. LinearRegression：fit 之後的 coef_ 和 intercept_ ==")
s = pd.read_csv("story/data/factory/stream/detections.csv")
s["ts"] = pd.to_datetime(s["ts"])
h = s.drop_duplicates().set_index("ts")["confidence"].resample("1h").count().rename("count").to_frame()
h["last_hour"] = h["count"].shift(1)
h = h.dropna()
train, test = h[h.index < "2026-01-09"], h[h.index >= "2026-01-09"]
print(f"每小時一列：學 {len(train)} 小時（01-05～01-08）、考 {len(test)} 小時（01-09 起）")
model = LinearRegression().fit(train[["last_hour"]], train["count"])
print(f"coef_ = {model.coef_.round(3)}、intercept_ = {model.intercept_:.3f}")
x = 50
print(f"上一小時 {x} 筆 → predict {model.predict(pd.DataFrame({'last_hour': [x]}))[0]:.3f}"
      f"；手算 {model.coef_[0]:.3f} × {x} + {model.intercept_:.3f} = {model.coef_[0] * x + model.intercept_:.3f}")

print("\n== 5. 和兩條 baseline 比（test，MAE 越小越好）==")
rows = [("平均：一律猜訓練期每小時的平均", [train["count"].mean()] * len(test)),
        ("上一小時：直接照抄上一小時", test["last_hour"]),
        ("LinearRegression：用上一小時", model.predict(test[["last_hour"]]))]
for name, pred in rows:
    print(f"MAE {mean_absolute_error(test['count'], pred):6.3f}  {name}")

print("\n== 6. 多一個落後特徵：昨天同一小時（shift(24)）==")
h2 = h.copy()
h2["same_hour_yesterday"] = h2["count"].shift(24)
h2 = h2.dropna()
cols = ["last_hour", "same_hour_yesterday"]
train2, test2 = h2[h2.index < "2026-01-09"], h2[h2.index >= "2026-01-09"]
model2 = LinearRegression().fit(train2[cols], train2["count"])
print("係數：", ", ".join(f"{c} {w:+.3f}" for c, w in zip(cols, model2.coef_)), f"；intercept {model2.intercept_:+.3f}")
print(f"MAE {mean_absolute_error(test2['count'], model2.predict(test2[cols])):.3f}（學 {len(train2)} 小時：前 24 小時沒有「昨天」，被 dropna 拿掉）")
