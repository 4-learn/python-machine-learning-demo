"""19｜偵測框 → Event JSON（全課的跨節介面，21 節的 Mock 模式直接讀這裡寫出的檔案）。

    python 19-detection-eval/to_events.py

規則：只有 head（沒戴）類別、且 confidence >= 偵測門檻才產生事件。
門檻讀 config/notify.yaml 的 camera.min_confidence，不寫死在程式裡。
"""
import json
from datetime import datetime, timedelta

import yaml
from ultralytics import YOLO

from common import ROOT
from detlib import HEAD, SCENES, WEIGHTS

CONFIG = yaml.safe_load((ROOT / "config" / "notify.yaml").read_text(encoding="utf-8"))["camera"]
OUT = ROOT / "19-detection-eval" / "events.json"

# 攝影機設定：哪支攝影機在哪個工地、哪一區（實務上來自設備清單，不是模型）
CAMERAS = {"cam1": ("A", "出入口"), "cam2": ("B", "物料區"), "cam3": ("C", "鋼筋區"), "cam4": ("D", "高架作業區")}
# 課程用的「影格」：哪張圖、哪支攝影機、何時拍的、追蹤器給的持續秒數與人員（本課不實作追蹤）
FRAMES = [
    ("test_000", "cam3", "2026-01-19 08:30:15", [(42.0, "W012")]),
    ("test_002", "cam1", "2026-01-19 08:41:02", []),
    ("test_014", "cam4", "2026-01-19 09:05:47", [(31.5, "W027"), (8.0, "W033")]),
    ("test_018", "cam2", "2026-01-19 10:12:30", [(66.0, "W041"), (12.5, "W008")]),
]
TZ = "+08:00"


def to_event(frame, camera, when, i, conf, box, track):
    site, zone = CAMERAS[camera]
    t = datetime.fromisoformat(when)
    duration, worker = track
    return {
        "event_id": f"{site}{camera[-1]}-{t:%Y%m%d-%H%M%S}-{i}",
        "camera_id": camera,
        "site": site,
        "zone": zone,
        "event_time": t.isoformat() + TZ,
        "class": "head",
        "confidence": round(conf, 3),
        "bbox": [round(v, 1) for v in box],
        "duration_sec": duration,
        "worker_id": worker,
        "detector": {"weights": "models/helmet-yolo11n.pt", "min_confidence": CONFIG["min_confidence"]},
    }


model = YOLO(WEIGHTS)
events, log = [], []
for frame, camera, when, tracks in FRAMES:
    r = model.predict(SCENES / "images" / "test" / f"{frame}.jpg", imgsz=320, conf=0.01, verbose=False)[0]
    boxes = sorted(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()), key=lambda b: b[2][0])
    heads = [(c, b) for k, c, b in boxes if int(k) == HEAD and c >= CONFIG["min_confidence"]]
    for i, ((conf, box), track) in enumerate(zip(heads, tracks)):
        events.append(to_event(frame, camera, when, i, conf, box, track))
    log.append({"frame": frame, "camera": camera, "框": len(boxes),
                "helmet": sum(int(k) == 0 and c >= CONFIG["min_confidence"] for k, c, _ in boxes),
                "head≥門檻": len(heads), "低於門檻": sum(c < CONFIG["min_confidence"] for _, c, _ in boxes)})

print(f"== 5. 偵測框 → 事件（門檻 {CONFIG['min_confidence']} 讀自 config/notify.yaml）==")
for row in log:
    print(row)
print(f"\n事件 {len(events)} 筆，第一筆：")
print("{")
items = list(events[0].items())
for n, (k, v) in enumerate(items):
    print(f'  "{k}": {json.dumps(v, ensure_ascii=False)}' + ("," if n < len(items) - 1 else ""))
print("}")

OUT.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"\n已寫出 19-detection-eval/events.json：{len(events)} 筆")
print("event_id：", [e["event_id"] for e in events])
