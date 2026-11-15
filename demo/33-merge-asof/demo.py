"""33｜pandas merge_asof：把兩個時間對不齊的來源，對到「最近的那一筆」。"""
import pandas as pd

# 左邊：每分鐘一格的偵測表（31 節的形狀）；右邊：照度計，大約每 10 秒一筆，時間對不上整分，還掉了幾筆
minutes = pd.DataFrame({"minute": pd.date_range("2026-01-05 08:00", periods=6, freq="1min"),
                        "detections": [2, 1, 0, 0, 0, 1]})
sensor = pd.DataFrame({
    "ts": pd.to_datetime(["2026-01-05 08:00:57", "2026-01-05 07:59:52", "2026-01-05 08:01:41",
                          "2026-01-05 08:01:58", "2026-01-05 08:02:05", "2026-01-05 08:05:20"]),
    "lux": [686.0, 690.0, 640.0, 634.0, 610.0, 652.0],
})

print("== 1. merge：時間要一模一樣才對得上 ==")
exact = minutes.merge(sensor, left_on="minute", right_on="ts", how="left")
print(exact.to_string(index=False))
print(f"對到照度的格子：{int(exact['lux'].notna().sum())} / {len(exact)}")

print("\n== 2. merge_asof 之前要先排序 ==")
try:
    pd.merge_asof(minutes, sensor, left_on="minute", right_on="ts")
except ValueError as e:
    print("ValueError：", e)
sensor = sensor.sort_values("ts")

print("\n== 3. direction：往回找、往後找、找最近 ==")
out = minutes.copy()
for direction in ["backward", "forward", "nearest"]:
    m = pd.merge_asof(minutes, sensor, left_on="minute", right_on="ts", direction=direction)
    out[direction] = m["ts"].dt.strftime("%H:%M:%S")
print(out.to_string(index=False))

print("\n== 4. tolerance：太久以前的讀值，不如沒有 ==")
m = pd.merge_asof(minutes, sensor, left_on="minute", right_on="ts",
                  direction="backward", tolerance=pd.Timedelta("1min"))
m["隔了多久"] = m["minute"] - m["ts"]
print(m.to_string(index=False))

print("\n== 5. by：兩個地點各有自己的照度計，只在同一個地點裡找 ==")
left = pd.DataFrame({"minute": pd.to_datetime(["2026-01-05 08:01", "2026-01-05 08:01", "2026-01-05 08:02"]),
                     "location": ["aisle_01", "dock_01", "dock_01"]})
right = pd.DataFrame({"ts": pd.to_datetime(["2026-01-05 08:00:40", "2026-01-05 08:00:55", "2026-01-05 08:01:50"]),
                      "location": ["dock_01", "aisle_01", "aisle_01"], "lux": [120.0, 686.0, 640.0]})
no_by = pd.merge_asof(left, right.drop(columns="location"), left_on="minute", right_on="ts")
with_by = pd.merge_asof(left, right, left_on="minute", right_on="ts", by="location")
print(left.assign(不加by=no_by["lux"], 加by=with_by["lux"]).to_string(index=False))

print("\n== 6. 真的資料：健康線的床墊壓力（每分鐘一筆、掉了 2%）==")
s = pd.read_csv("story/data/care/stream/sensor.csv")
s["ts"] = pd.to_datetime(s["ts"])
grid = pd.DataFrame({"minute": pd.date_range(s["ts"].min(), s["ts"].max(), freq="1min")})
exact = grid.merge(s, left_on="minute", right_on="ts", how="left")
for tol in ["1min", "5min"]:
    a = pd.merge_asof(grid, s.sort_values("ts"), left_on="minute", right_on="ts", tolerance=pd.Timedelta(tol))
    print(f"tolerance {tol}：對不到 {int(a['value'].isna().sum())} 格")
print(f"merge（時間相等）：對不到 {int(exact['value'].isna().sum())} 格；一共 {len(grid)} 格")
