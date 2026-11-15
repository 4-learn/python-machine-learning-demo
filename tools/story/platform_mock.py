"""API 課平台的「替身」：只做 ML 課需要的那一小段（POST /events、GET /alerts）。

規則和 API 課 v0.12 一樣：
  - 沒帶正確的 X-API-Key → 401
  - 事件類型不在設定檔 → 400
  - 信心值 ≥ 0.8 開 open、0.5–0.8 開 pending_review、< 0.5 只記錄
  - 事件的 detail 裡有 judge.decision 時，改聽判斷器的（第 23 節）

真正的平台在 API 課（FastAPI＋PostgreSQL）。要改送到你自己的平台：
    PLATFORM_URL=http://localhost:8000 python detector.py
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEVICE_KEY = "dev-device"


def make_server(profile: dict, port: int = 0) -> ThreadingHTTPServer:
    events, alerts = [], []

    def decide(event):
        judge = (event.get("detail") or {}).get("judge")
        if judge:
            return {"auto_alert": "open", "require_review": "pending_review", "log_only": None}[judge["decision"]]
        c = event["confidence"]
        return "open" if c >= 0.8 else "pending_review" if c >= 0.5 else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):   # 不印存取紀錄
            pass

        def reply(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/alerts"):
                return self.reply(200, alerts)
            if self.path == "/health":
                return self.reply(200, {"status": "ok", "customer": profile["customer"]})
            self.reply(404, {"detail": "Not Found"})

        def do_POST(self):
            if self.path != "/events":
                return self.reply(404, {"detail": "Not Found"})
            if self.headers.get("X-API-Key") != DEVICE_KEY:
                return self.reply(401, {"detail": "缺少或錯誤的 API Key"})
            event = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if event.get("event_type") not in profile["event_types"]:
                return self.reply(400, {"detail": f"不支援的事件類型：{event.get('event_type')}"})
            events.append(event)
            status = decide(event)
            body = {"status": "received", "event_id": len(events), "alert_id": None, "alert_status": None}
            if status:
                alerts.append({"id": len(alerts) + 1, "event_id": len(events), "event_type": event["event_type"],
                               "location": event["location"], "status": status})
                body.update(alert_id=len(alerts), alert_status=status)
            self.reply(200, body)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def start(profile: dict) -> str:
    """在背景啟動替身平台，回傳它的網址。"""
    server = make_server(profile)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}"
