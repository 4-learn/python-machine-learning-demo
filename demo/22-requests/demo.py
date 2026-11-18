"""22｜requests：把偵測框變成平台格式的事件，POST 到平台；平台拒收、沒開時也不會當掉。"""
import json
import socket
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "story"))
import platform_mock  # noqa: E402  告警平台的替身

profile = json.loads((ROOT / "story" / "profiles" / "factory.json").read_text(encoding="utf-8"))

# 偵測器在一張照片上找到的框：cls 1＝head（沒戴安全帽），0＝helmet
boxes = [
    {"cls": 1, "conf": 0.962, "xyxy": [120.0, 40.0, 152.0, 74.0]},
    {"cls": 0, "conf": 0.981, "xyxy": [200.0, 52.0, 230.0, 80.0]},
    {"cls": 1, "conf": 0.412, "xyxy": [20.0, 60.0, 44.0, 90.0]},
]
photo = {"image": "test_001.jpg", "source": "camera_04", "location": "dock_01",
         "occurred_at": "2026-01-05 01:37:00", "duration_seconds": 77.8, "light_lux": 70, "person_id": "W004"}

print("== 1. 偵測框 → 事件 JSON（平台格式） ==")
hits = [b for b in boxes if b["cls"] == 1 and b["conf"] >= profile["detector_conf"]]
event = {"event_type": "no_helmet", "location": photo["location"], "source": photo["source"],
         "confidence": max(b["conf"] for b in hits), "occurred_at": photo["occurred_at"].replace(" ", "T"),
         "detail": {"duration_seconds": photo["duration_seconds"], "light_lux": photo["light_lux"],
                    "person_id": photo["person_id"], "image": photo["image"]}}
print(f"{len(boxes)} 個框，類別 1 且信心值 ≥ {profile['detector_conf']} 的：{len(hits)} 個")
print(json.dumps(event, ensure_ascii=False, indent=1))

print("\n== 2. requests.post：送出去，看狀態碼和回應 ==")
URL = platform_mock.start(profile)          # 換成 "http://localhost:8000" 就是送到你自己架的平台
KEY = {"X-API-Key": "dev-device"}
r = requests.post(f"{URL}/events", json=event, headers=KEY, timeout=5)
print(r.status_code, r.json())

print("\n== 3. 平台拒收的兩種情況 ==")
r = requests.post(f"{URL}/events", json=event, headers={"X-API-Key": "wrong-key"}, timeout=5)
print(f"錯的 API Key   → {r.status_code} {r.json()['detail']}")
r = requests.post(f"{URL}/events", json={**event, "event_type": "smoking"}, headers=KEY, timeout=5)
print(f"不支援的事件類型 → {r.status_code} {r.json()['detail']}")

print("\n== 4. 包成一個不會讓偵測器當掉的 send() ==")


def send(url, event, api_key="dev-device"):
    try:
        r = requests.post(f"{url}/events", json=event, headers={"X-API-Key": api_key}, timeout=5)
    except requests.exceptions.RequestException as e:
        return None, {"detail": type(e).__name__}
    return r.status_code, r.json()


probe = socket.socket()                     # 借一個空的 port，關掉，當成「沒開的平台」
probe.bind(("127.0.0.1", 0))
closed_port = probe.getsockname()[1]
probe.close()
print(f"正常     → {send(URL, event)}")
print(f"錯的 key → {send(URL, event, api_key='wrong-key')}")
print(f"平台沒開 → {send(f'http://127.0.0.1:{closed_port}', event)}")
