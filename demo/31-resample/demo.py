"""31｜pandas resample：時間不規則、有重複的偵測串流，整理成每分鐘一格。"""
import pandas as pd

# 偵測器在 08:00–08:05 送出的 9 筆偵測（時間是文字、間隔不規則，第 4 筆被重送了一次）
raw = pd.DataFrame({
    "ts": ["2026-01-05 08:00:12.5", "2026-01-05 08:00:47.1", "2026-01-05 08:01:05.9",
           "2026-01-05 08:01:30.0", "2026-01-05 08:01:30.0", "2026-01-05 08:04:02.3",
           "2026-01-05 08:04:58.8", "2026-01-05 08:05:00.0", "2026-01-05 08:00:30.2"],
    "source": ["camera_01", "camera_02", "camera_01", "camera_03", "camera_03",
               "camera_01", "camera_02", "camera_01", "camera_03"],
    "confidence": [0.512, 0.874, 0.433, 0.691, 0.691, 0.388, 0.925, 0.604, 0.557],
})

print("== 1. pd.to_datetime：文字變成時間 ==")
print("轉換前：", raw["ts"].dtype, repr(raw["ts"].iloc[0]))
raw["ts"] = pd.to_datetime(raw["ts"])
print("轉換後：", raw["ts"].dtype, repr(raw["ts"].iloc[0]))
print("可以相減：", raw["ts"].iloc[1] - raw["ts"].iloc[0])

print("\n== 2. drop_duplicates：同一筆送了兩次 ==")
print(f"重複的列：{int(raw.duplicated().sum())} 筆")
stream = raw.drop_duplicates()
print(f"去掉之後：{len(raw)} → {len(stream)} 筆")

print("\n== 3. set_index＋sort_index：時間當索引、照時間排好 ==")
stream = stream.set_index("ts").sort_index()
print(stream.to_string())
gaps = stream.index.to_series().diff()
print(f"相鄰兩筆的時間差：最短 {gaps.min()}、最長 {gaps.max()}")

print("\n== 4. resample('1min').agg：每分鐘一格 ==")
minutes = stream.resample("1min").agg({"confidence": ["count", "max"]})
minutes.columns = ["detections", "max_conf"]
print(minutes.to_string())

print("\n== 5. 空的那幾分鐘：count 是 0，max 和 mean 是 NaN ==")
empty = minutes[minutes["detections"] == 0]
print(f"沒有偵測的分鐘：{len(empty)} 格（{', '.join(empty.index.strftime('%H:%M'))}）")
print(f"max_conf 平均（跳過 NaN）：{minutes['max_conf'].mean():.3f}")
print(f"max_conf 先補 0 再平均  ：{minutes['max_conf'].fillna(0).mean():.3f}")

print("\n== 6. 換成真的串流：換一個 freq，就換一種格子 ==")
real = pd.read_csv("story/data/factory/stream/detections.csv")
real["ts"] = pd.to_datetime(real["ts"])
real = real.drop_duplicates().set_index("ts").sort_index()
print(f"{len(real)} 筆；每分鐘 {len(real.resample('1min').size())} 格、每小時 {len(real.resample('1h').size())} 格")
print(real.resample("6h").size().head(4).rename("detections").to_string())
