"""21｜真實模式：對 19 節同樣的 4 張影格跑 YOLO，確認產生的 Event JSON 與 Mock 檔完全相同。

    python 21-pipeline/run_yolo.py         # CPU 約 10 秒
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import NotifyPipeline, detect_events, load_mock_events  # noqa: E402

from common import DATA  # noqa: E402

TEST = DATA / "scenes" / "images" / "test"
FRAMES = [  # 與 19 節 to_events.py 相同：哪張圖、哪支攝影機、何時、追蹤器給的（持續秒數, 人員）
    (TEST / "test_000.jpg", "cam3", "2026-01-19 08:30:15", [(42.0, "W012")]),
    (TEST / "test_002.jpg", "cam1", "2026-01-19 08:41:02", []),
    (TEST / "test_014.jpg", "cam4", "2026-01-19 09:05:47", [(31.5, "W027"), (8.0, "W033")]),
    (TEST / "test_018.jpg", "cam2", "2026-01-19 10:12:30", [(66.0, "W041"), (12.5, "W008")]),
]

print("== 6. 真實模式：YOLO → Event JSON ==")
live = detect_events(FRAMES)
mock = load_mock_events()
print(f"YOLO 產生 {len(live)} 筆事件；Mock 檔 {len(mock)} 筆")
print("兩者完全相同：", live == mock)

pipe = NotifyPipeline()
same = pipe.run(live)[["event_id", "action"]].equals(pipe.run(mock)[["event_id", "action"]])
print("後面的決定也相同：", same)
