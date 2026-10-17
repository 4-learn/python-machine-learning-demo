"""21｜事件管線：Event JSON → 補情境欄位 → 模型機率 → 核准門檻 → 政策規則 → 決定（附版本）。

這支檔案可以被 import（23 節沿用）：

    import sys; sys.path.insert(0, "21-pipeline")
    from pipeline import NotifyPipeline, load_mock_events
    pipe = NotifyPipeline()
    decisions = pipe.run(load_mock_events())

兩條路徑：
- 即時路徑：一筆 Event JSON 進來 → decide() → 毫秒內決定要不要通知。
- 離線路徑：一整批已有欄位的事件表 → score_table() → 統計、監控（22 節）、驗收（23 節）。
離線路徑**不會**改設定檔；門檻只能經人工核准後改 config/notify.yaml。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml

from common import FEATURES, ROOT, load_events

HERE = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "notify.yaml"
RULES = HERE / "rules.yaml"
SITE = HERE / "site.yaml"
MOCK_EVENTS = HERE / "mock_events.json"


def read_yaml(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sha256_12(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def shift_of(hour: int) -> str:
    """與 events.csv 相同的班別定義。"""
    if 6 <= hour < 14:
        return "早班"
    if 14 <= hour < 22:
        return "中班"
    return "夜班"


# ---------------------------------------------------------------- 1. 偵測（真的 YOLO 或 Mock）
def load_mock_events(path=MOCK_EVENTS) -> list[dict]:
    """Mock 模式：不載入 YOLO，直接讀事先存好的 Event JSON（19 節產生的格式）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def detect_events(frames, config=CONFIG, site=SITE) -> list[dict]:
    """真實模式：對影格跑 YOLO，轉成 Event JSON。frames = [(圖檔, 攝影機, 時間, [(持續秒數, 人員)])]。
    只有 head（沒戴）且信心 >= camera.min_confidence 才產生事件（19 節的規則）。"""
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO  # 只有真實模式才需要；Mock 模式與測試不必載入

    cam_cfg = read_yaml(config)["camera"]
    cameras = read_yaml(site)["cameras"]
    model = YOLO(ROOT / cam_cfg["detector"])
    events = []
    for image, camera, when, tracks in frames:
        r = model.predict(image, imgsz=320, conf=0.01, verbose=False)[0]
        boxes = sorted(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()), key=lambda b: b[2][0])
        heads = [(c, b) for k, c, b in boxes if r.names[int(k)] == "head" and c >= cam_cfg["min_confidence"]]
        for i, ((conf, box), (duration, worker)) in enumerate(zip(heads, tracks)):
            t = pd.Timestamp(when, tz="Asia/Taipei")
            cam = cameras[camera]
            events.append({
                "event_id": f"{cam['site']}{camera[-1]}-{t:%Y%m%d-%H%M%S}-{i}",
                "camera_id": camera, "site": cam["site"], "zone": cam["zone"],
                "event_time": t.isoformat(), "class": "head", "confidence": round(conf, 3),
                "bbox": [round(v, 1) for v in box], "duration_sec": duration, "worker_id": worker,
                "detector": {"weights": cam_cfg["detector"], "min_confidence": cam_cfg["min_confidence"]},
            })
    return events


# ---------------------------------------------------------------- 2–5. 判斷
class NotifyPipeline:
    """讀三份設定：模型與門檻（config/notify.yaml）、政策規則（rules.yaml）、攝影機（site.yaml）。"""

    def __init__(self, config=CONFIG, rules=RULES, site=SITE, history: pd.DataFrame | None = None):
        cfg = read_yaml(config)
        model_file = ROOT / cfg["model"]["file"]
        actual = sha256_12(model_file)
        if actual != cfg["model"]["sha256_12"]:
            raise ValueError(f"模型檔 {model_file.name} 的 sha256_12 是 {actual}，設定檔寫的是 {cfg['model']['sha256_12']}")
        self.model = joblib.load(model_file)
        self.threshold = float(cfg["threshold"]["value"])
        rule_cfg = read_yaml(rules)
        self.rules = rule_cfg["rules"]
        self.cooldown = pd.Timedelta(seconds=rule_cfg.get("cooldown_sec", 0))
        self.site = read_yaml(site)
        self.history = load_events() if history is None else history
        self.versions = {
            "model.version": cfg["model"]["version"],
            "model.sha256_12": actual,
            "threshold.value": self.threshold,
            "threshold.approved_by": cfg["threshold"]["approved_by"],
            "threshold.approved_at": str(cfg["threshold"]["approved_at"]),
            "rules.version": rule_cfg["version"],
        }

    # ---- 2. 補情境欄位：Event JSON 只有偵測器知道的事，其他欄位要查
    def features(self, event: dict) -> dict:
        t = pd.Timestamp(event["event_time"])
        hour = t.hour
        shift = shift_of(hour)
        cam = self.site["cameras"][event["camera_id"]]
        worker = self.history[self.history.worker_id == event["worker_id"]]
        past = worker[(worker.needs_notify == 1) & (worker.event_time < t)
                      & (worker.event_time >= t - pd.Timedelta(days=30))]
        return {
            "confidence": event["confidence"], "duration_sec": event["duration_sec"],
            "light_lux": float(cam["light_lux"][shift]), "people_nearby": int(self.site["people_nearby"]),
            "tenure_days": int(worker.tenure_days.iloc[-1]) if len(worker) else 0,
            "prior_violations_30d": len(past), "hour": hour,
            "zone": event["zone"], "shift": shift, "site": event["site"],
        }

    # ---- 3. 模型機率
    def proba(self, rows: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict_proba(rows[FEATURES])[:, 1], index=rows.index)

    # ---- 4. 政策規則：由上往下，第一條符合就生效
    def match_rule(self, row: dict) -> dict:
        for rule in self.rules:
            when = rule.get("when", {})
            if "zone" in when and row["zone"] not in when["zone"]:
                continue
            if "model_notify" in when and row["model_notify"] != when["model_notify"]:
                continue
            if "min_prior_violations_30d" in when and row["prior_violations_30d"] < when["min_prior_violations_30d"]:
                continue
            return rule
        raise ValueError("沒有任何規則符合；rules.yaml 最後一條應是沒有條件的預設")

    def apply_rules(self, table: pd.DataFrame) -> pd.DataFrame:
        table = table.copy()
        table["model_notify"] = table["proba"] >= self.threshold
        matched = [self.match_rule(r) for r in table.to_dict("records")]
        table["rule"] = [m["name"] for m in matched]
        table["action"] = [m["action"] for m in matched]
        return table

    # ---- 5. 冷卻：同一人、同一工地在冷卻時間內只通知一次（其餘標成 merged）
    def apply_cooldown(self, table: pd.DataFrame) -> pd.DataFrame:
        table = table.sort_values("event_time", kind="stable").copy()
        last: dict = {}
        actions = []
        for key, t, action in zip(zip(table.worker_id, table.site), table.event_time, table.action):
            if action == "notify" and key in last and t - last[key] <= self.cooldown:
                actions.append("merged")
                continue
            if action == "notify":
                last[key] = t
            actions.append(action)
        table["action"] = actions
        return table.sort_index()

    def stamp(self, table: pd.DataFrame) -> pd.DataFrame:
        for k, v in self.versions.items():
            table[k] = v
        return table

    # ---- 即時路徑：Event JSON 進、決定出
    def run(self, events: list[dict]) -> pd.DataFrame:
        rows = pd.DataFrame([{**self.features(e), "event_id": e["event_id"], "worker_id": e["worker_id"],
                              "event_time": pd.Timestamp(e["event_time"])} for e in events])
        rows["proba"] = self.proba(rows).round(3)
        return self.stamp(self.apply_cooldown(self.apply_rules(rows)))

    def decide(self, event: dict) -> dict:
        return self.run([event]).iloc[0].to_dict()

    # ---- 離線路徑：已經有 FEATURES 欄位的事件表（events.csv、events-new-weeks.csv）
    def score_table(self, df: pd.DataFrame, cooldown: bool = True) -> pd.DataFrame:
        out = df.copy()
        out["proba"] = self.proba(out).round(3)
        out = self.apply_rules(out)
        if cooldown:
            out = self.apply_cooldown(out)
        return self.stamp(out)


def summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """統計：每天各種決定各幾筆（舊版「規則 → 統計」的統計段）。"""
    day = pd.to_datetime(decisions.event_time).dt.strftime("%m-%d")
    return pd.crosstab(day, decisions.action).rename_axis(index="日期", columns=None)
