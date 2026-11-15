"""23｜Workshop 2：把判斷器的決定帶進事件的 detail，平台照它開告警；再和事後答案比一次。"""
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "story"))
import platform_mock  # noqa: E402  API 課平台的替身：detail 裡有 judge.decision 時改聽判斷器的

profile = json.loads((ROOT / "story" / "profiles" / "factory.json").read_text(encoding="utf-8"))
KEY = {"X-API-Key": "dev-device"}

# 偵測器在 5 張照片上都看到「沒戴安全帽」，信心值都很高。should_alert 是事後答案，只拿來對照、不送出去。
photos = pd.DataFrame({
    "image": ["p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg", "p5.jpg"],
    "location": ["aisle_01", "rest_01", "dock_01", "aisle_02", "aisle_01"],
    "confidence": [0.968, 0.979, 0.964, 0.987, 0.969],
    "duration_seconds": [74.0, 28.2, 13.1, 96.7, 7.2],
    "should_alert": [1, 0, 0, 1, 0],
})
# 判斷器（15 節的 Judge.decide）對這 5 筆的回答；這裡手打，Workshop 用真的 Judge
judged = [{"decision": "auto_alert", "score": 0.998, "model": "3b509d5e93ae"},
          {"decision": "require_review", "score": 0.547, "model": "3b509d5e93ae"},
          {"decision": "require_review", "score": 0.633, "model": "3b509d5e93ae"},
          {"decision": "auto_alert", "score": 0.997, "model": "3b509d5e93ae"},
          {"decision": "log_only", "score": 0.012, "model": "3b509d5e93ae"}]


def event_of(row):
    return {"event_type": "no_helmet", "location": row["location"], "source": "camera_01",
            "confidence": row["confidence"], "occurred_at": "2026-01-06T18:02:00",
            "detail": {"duration_seconds": row["duration_seconds"], "image": row["image"]}}


def post_all(events):
    url = platform_mock.start(profile)           # 每次開一個新的平台，告警從 0 開始
    return [requests.post(f"{url}/events", json=e, headers=KEY, timeout=5).json()["alert_status"] or "（只記錄）"
            for e in events]


print("== 1. 只送偵測結果：平台照信心值 0.8／0.5 ==")
events = [event_of(row) for _, row in photos.iterrows()]
photos["平台規則"] = post_all(events)
print(photos[["image", "location", "confidence", "平台規則"]].to_string(index=False))

print("\n== 2. 判斷器的決定放進 detail.judge ==")
with_judge = [{**e, "detail": {**e["detail"], "judge": j}} for e, j in zip(events, judged)]
print(json.dumps(with_judge[1], ensure_ascii=False))
photos["判斷器"] = post_all(with_judge)
print(photos[["image", "location", "duration_seconds", "平台規則", "判斷器"]].to_string(index=False))

print("\n== 3. {**e} 只複製外層：detail 還是同一個 ==")
shallow = {**events[0]}
shallow["detail"]["judge"] = judged[0]
print(f"改了複製品的 detail，原本的事件也有 judge：{'judge' in events[0]['detail']}")
del events[0]["detail"]["judge"]

print("\n== 4. 和事後答案比 ==")


def compare(status, should):
    called, review = set(photos.index[status == "open"]), set(photos.index[status == "pending_review"])
    yes = set(photos.index[should == 1])
    return {"馬上通知": len(called), "其中該叫": len(called & yes), "待確認": len(review),
            "漏掉": len(yes - called - review)}


print(pd.DataFrame([{"方法": "平台規則", **compare(photos["平台規則"], photos["should_alert"])},
                    {"方法": "偵測器＋判斷器", **compare(photos["判斷器"], photos["should_alert"])}]).to_string(index=False))

print("\n== 5. 判斷器改變了哪些決定 ==")
print(photos[photos["平台規則"] != photos["判斷器"]][["image", "location", "duration_seconds", "平台規則", "判斷器",
                                                    "should_alert"]].to_string(index=False))
