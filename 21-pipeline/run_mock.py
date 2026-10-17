"""21｜Mock 模式跑一次管線：不載入 YOLO，讀 19 節存好的 Event JSON。

    python 21-pipeline/run_mock.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import NotifyPipeline, load_mock_events, summary  # noqa: E402

from common import load_events  # noqa: E402

pipe = NotifyPipeline()
events = load_mock_events()

print("== 1. 這次判斷用的版本（每筆決定都會帶著）==")
for k, v in pipe.versions.items():
    print(f"{k:22s} {v}")

print("\n== 2. Event JSON 只有偵測器知道的事，其他欄位要查 ==")
e = events[0]
print("Event JSON：", {k: e[k] for k in ["event_id", "camera_id", "zone", "event_time", "confidence", "duration_sec", "worker_id"]})
print("補成模型要的一列：", pipe.features(e))

print("\n== 3. 即時路徑：5 筆事件的決定 ==")
d = pipe.run(events)
cols = ["event_id", "zone", "duration_sec", "prior_violations_30d", "proba", "rule", "action"]
print(d[cols].to_string(index=False))

print("\n== 4. 同輸入同輸出 ==")
again = NotifyPipeline().run(load_mock_events())
print("重新載入、再跑一次，決定完全相同：", again[cols].equals(d[cols]))
start = time.perf_counter()
pipe.decide(events[0])
print("1 筆決定耗時 < 100 毫秒：", time.perf_counter() - start < 0.1)

print("\n== 5. 離線路徑：上線第一週（01-12～01-18）每天的決定 ==")
week = load_events("data/events-new-weeks.csv")
week = week[week.event_time < "2026-01-19"]
offline = pipe.score_table(week)
table = summary(offline)
print(table.to_string())
print(f"一週 {len(offline)} 筆事件；通知 {(offline.action == 'notify').sum()} 次（平均每天 {(offline.action == 'notify').sum() / 7:.1f} 次）；"
      f"冷卻合併 {(offline.action == 'merged').sum()} 筆")
print("每一列都有版本欄：", offline[list(pipe.versions)].notna().all().all())
