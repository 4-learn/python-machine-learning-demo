"""32｜pandas rolling：只看過去的時間特徵（closed='left'、center=True 偷看未來、時間視窗 vs 筆數視窗）。"""
import pandas as pd

# 已經 resample 好的每分鐘偵測數（31 節的做法），08:00–08:09
idx = pd.date_range("2026-01-05 08:00", periods=10, freq="1min")
minutes = pd.DataFrame({"detections": [2, 0, 1, 0, 0, 3, 0, 0, 4, 1]}, index=idx)

print("== 1. rolling('3min').sum()：每一格往回看 3 分鐘 ==")
m = minutes.copy()
m["sum_3min"] = m["detections"].rolling("3min").sum()
print(m.to_string())

print("\n== 2. closed='left'：不含這一分鐘自己 ==")
m["past_3min"] = m["detections"].rolling("3min", closed="left").sum()
print(m.to_string())

print("\n== 3. center=True：一半的視窗在未來 ==")
m["center_3min"] = m["detections"].rolling("3min", center=True).sum()
print(m[["detections", "past_3min", "center_3min"]].iloc[3:9].to_string())

print("\n== 4. 筆數視窗 vs 時間視窗：在不規則的原始串流上 ==")
raw = pd.DataFrame({"confidence": [0.512, 0.557, 0.874, 0.433, 0.691, 0.388, 0.925, 0.604]},
                   index=pd.to_datetime(["2026-01-05 08:00:12", "2026-01-05 08:00:30", "2026-01-05 08:00:47",
                                         "2026-01-05 08:01:05", "2026-01-05 08:01:30", "2026-01-05 08:04:02",
                                         "2026-01-05 08:04:58", "2026-01-05 08:09:00"]))
raw["n_last3"] = raw["confidence"].rolling(3).count()
raw["n_3min"] = raw["confidence"].rolling("3min").count()
raw["span_last3"] = raw.index.to_series().diff(2)
print(raw.to_string())

print("\n== 5. 真的串流（工廠線一週）：三種視窗，依「這一分鐘有沒有偵測」分組平均 ==")
s = pd.read_csv("story/data/factory/stream/detections.csv")
s["ts"] = pd.to_datetime(s["ts"])
per_min = s.drop_duplicates().set_index("ts").sort_index().resample("1min").size()
table = pd.DataFrame({"這一分鐘有偵測": (per_min > 0).astype(int),
                      "過去 10 分鐘（closed='left'）": per_min.rolling("10min", closed="left").sum().fillna(0),
                      "含這一分鐘（預設）": per_min.rolling("10min").sum(),
                      "前後 10 分鐘（center=True）": per_min.rolling("10min", center=True).sum()})
print(table.groupby("這一分鐘有偵測").mean().round(3).to_string())
