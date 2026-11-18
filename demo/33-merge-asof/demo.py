"""33｜pandas merge_asof：把兩個時間對不齊的來源，對到「這一分鐘以前、最近的那一筆」。"""
import pandas as pd

# 左邊：每分鐘一格的偵測表（31 節的形狀）；右邊：照度計，時間對不上整分，還掉了幾筆、順序也亂了
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

print("\n== 2. merge_asof：先排序，再往回找最近的一筆 ==")
sensor = sensor.sort_values("ts")
m = pd.merge_asof(minutes, sensor, left_on="minute", right_on="ts")
print(m.to_string(index=False))

print("\n== 3. tolerance：太久以前的讀值，不如沒有 ==")
m = pd.merge_asof(minutes, sensor, left_on="minute", right_on="ts",
                  direction="backward", tolerance=pd.Timedelta("1min"))
m["隔了多久"] = m["minute"] - m["ts"]
print(m.to_string(index=False))

print("\n== 4. 真的資料：健康線的床墊壓力（每分鐘一筆、掉了 2%）==")
s = pd.read_csv("story/data/care/stream/sensor.csv")
s["ts"] = pd.to_datetime(s["ts"])
grid = pd.DataFrame({"minute": pd.date_range(s["ts"].min(), s["ts"].max(), freq="1min")})
for tol in ["1min", "5min"]:
    a = pd.merge_asof(grid, s.sort_values("ts"), left_on="minute", right_on="ts", tolerance=pd.Timedelta(tol))
    print(f"tolerance {tol}：對不到 {int(a['value'].isna().sum())} 格")
exact = grid.merge(s, left_on="minute", right_on="ts", how="left")
print(f"merge（時間相等）：對不到 {int(exact['value'].isna().sum())} 格；一共 {len(grid)} 格")
