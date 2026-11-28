"""自己架的告警平台（FastAPI 版）：規則和 story/platform_mock.py 一樣。

啟動（在 story/platform 這個資料夾裡）：
    uvicorn main:app --port 8000

偵測器改送到這裡（另開一個終端機，在 story/my 裡）：
    PLATFORM_URL=http://localhost:8000 LINE=factory python detector.py
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException

LINE = os.getenv("LINE", "factory")
PROFILE = json.loads((Path(__file__).resolve().parent.parent / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
DEVICE_KEY = "dev-device"

app = FastAPI(title="告警平台")
events = []   # 收到的事件（沒有資料庫：程式一停就不見）
alerts = []   # 開出來的告警


def decide(event):
    """平台的分流規則：有判斷器的決定就聽判斷器，沒有就看信心值。"""
    judge = (event.get("detail") or {}).get("judge")
    if judge:
        return {"auto_alert": "open", "require_review": "pending_review", "log_only": None}[judge["decision"]]
    c = event["confidence"]
    if c >= 0.8:
        return "open"             # 馬上通知
    if c >= 0.5:
        return "pending_review"   # 待確認
    return None                   # 只記錄，不開告警


@app.post("/events")
def receive_event(event: dict, x_api_key: str = Header(default="")):
    # 規則 1：沒帶正確的 API Key → 401
    if x_api_key != DEVICE_KEY:
        raise HTTPException(status_code=401, detail="缺少或錯誤的 API Key")
    # 規則 2：事件類型不在客戶設定檔 → 400
    if event.get("event_type") not in PROFILE["event_types"]:
        raise HTTPException(status_code=400, detail=f"不支援的事件類型：{event.get('event_type')}")
    # 規則 3：收下事件，照信心值（或判斷器）分流
    events.append(event)
    status = decide(event)
    body = {"status": "received", "event_id": len(events), "alert_id": None, "alert_status": None}
    if status:
        alerts.append({"id": len(alerts) + 1, "event_id": len(events), "event_type": event["event_type"],
                       "location": event["location"], "status": status})
        body.update(alert_id=len(alerts), alert_status=status)
    return body


@app.get("/alerts")
def list_alerts():
    return alerts


@app.get("/health")
def health():
    return {"status": "ok", "customer": PROFILE["customer"]}
