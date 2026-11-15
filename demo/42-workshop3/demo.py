"""42｜Workshop 3：三種偵測器（ROI、骨架、物件偵測）送進同一個平台；新事件類型要先加進設定檔。"""
import base64
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "story"))
import platform_mock  # noqa: E402  API 課平台的替身：事件類型不在設定檔就回 400

profile = json.loads((ROOT / "story" / "profiles" / "factory.json").read_text(encoding="utf-8"))
KEY = {"X-API-Key": "dev-device"}
url = platform_mock.start(profile)


def post(event):
    r = requests.post(f"{url}/events", json=event, headers=KEY, timeout=5)
    return r.status_code, r.json()


# 三種偵測器各送一筆。信心值、分數都是手打的；Workshop 用真的 ROI、骨架模型和 YOLO。
roi_event = {"event_type": "aisle_blocked", "location": "aisle_01", "source": "ezcam_01", "confidence": 0.62,
             "occurred_at": "2026-01-19T09:12:00", "detail": {"roi_score": 0.074, "frame": "frame_012.jpg"}}
pose_event = {"event_type": "unsafe_lifting", "location": "aisle_01", "source": "pose_01", "confidence": 0.91,
              "occurred_at": "2026-01-19T10:00:00", "detail": {"seq_id": "SQ0042", "torso_angle": 62.4, "hip_drop": 0.31}}
yolo_event = {"event_type": "no_helmet", "location": "dock_01", "source": "camera_03", "confidence": 0.968,
              "occurred_at": "2026-01-19T10:05:00", "detail": {"duration_seconds": 74.0, "image": "test_001.jpg"}}

print("== 1. 設定檔現在收哪些事件類型 ==")
print(list(profile["event_types"]))



def send_all(pairs):
    rows = []
    for name, event in pairs:
        code, body = post(event)
        rows.append({"來源": name, "event_type": event["event_type"], "confidence": event["confidence"],
                     "HTTP": code, "平台回答": body.get("alert_status") or body.get("detail")})
    return pd.DataFrame(rows).to_string(index=False)


print("\n== 2. 三種事件送進同一個平台 ==")
print(send_all([("ROI", roi_event), ("骨架", pose_event), ("YOLO", yolo_event)]))

print("\n== 3. 新事件類型先加進設定檔，再送一次 ==")
profile["event_types"]["unsafe_lifting"] = {"severity": "high", "escalate_after_minutes": 10}
print(list(profile["event_types"]))
print(send_all([("骨架", pose_event)]))

print("\n== 4. 骨架事件只送分數與特徵，不送畫面 ==")
photo = (ROOT / "story" / "data" / "photos" / "factory-lifting.jpg").read_bytes()
with_photo = {**pose_event, "detail": {**pose_event["detail"], "image_base64": base64.b64encode(photo).decode()}}
print(f"只送特徵：{len(json.dumps(pose_event).encode()):>7} bytes")
print(f"附上畫面：{len(json.dumps(with_photo).encode()):>7} bytes")
print("detail：", json.dumps(pose_event["detail"], ensure_ascii=False))

print("\n== 5. 平台上的告警：依事件類型看 ==")
alerts = pd.DataFrame(requests.get(f"{url}/alerts", timeout=5).json())
print(alerts[["id", "event_type", "location", "status"]].to_string(index=False))
