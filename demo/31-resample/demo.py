"""31｜pandas resample：時間不規則、有重送的偵測串流，整理成每分鐘一格。"""
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

print("\n== 2. 去掉重送、照時間排好 ==")
stream = raw.drop_duplicates()
print(f"去掉重送：{len(raw)} → {len(stream)} 筆")
stream = stream.set_index("ts").sort_index()
print(stream.to_string())

print("\n== 3. resample('1min')：每分鐘一格 ==")
minutes = stream.resample("1min").agg({"confidence": ["count", "max"]})
minutes.columns = ["detections", "max_conf"]
print(minutes.to_string())

print("\n== 4. 空的格子：補 0 會讓平均變假 ==")
print(f"max_conf 平均（跳過 NaN）：{minutes['max_conf'].mean():.3f}")
print(f"max_conf 先補 0 再平均  ：{minutes['max_conf'].fillna(0).mean():.3f}")
