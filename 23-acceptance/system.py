"""23｜舊版作業的五個 class，改寫成「讀核准設定」的版本；判斷核心沿用 21 節的 NotifyPipeline。

    import sys; sys.path.insert(0, "23-acceptance")
    from system import DetectionPipeline
    decisions = DetectionPipeline(mock=True).process_frames(frames)

| class               | 做什麼                                   | 舊版                    |
| ------------------- | ---------------------------------------- | ----------------------- |
| SafetyDetector      | YOLO（或 Mock）→ head 框                 | 保留                    |
| EventConverter      | head 框 → Event JSON（19 節格式）        | 保留                    |
| ThresholdCalibrator | **讀**核准門檻；沒核准就拒絕             | 原本用 KMeans 算門檻    |
| RuleEngine          | 21 節的模型＋規則＋冷卻；pandas 統計     | 原本只看 confidence     |
| DetectionPipeline   | 串起來                                   | 原本每 N 張自動校準     |
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "21-pipeline"))
from pipeline import CONFIG, RULES, SITE, NotifyPipeline, read_yaml, summary  # noqa: E402

from common import DATA, load_events  # noqa: E402

MOCK_DETECTIONS = HERE / "detections.json"


class SafetyDetector:
    """回傳一張圖裡所有 head（沒戴）框，信心 >= camera.min_confidence，由左到右排。"""

    def __init__(self, config=CONFIG, mock: bool = False, mock_path=MOCK_DETECTIONS):
        cam = read_yaml(config)["camera"]
        self.weights, self.min_confidence = cam["detector"], cam["min_confidence"]
        self.mock = mock
        if mock:
            self.saved = json.loads(Path(mock_path).read_text(encoding="utf-8"))
        else:
            os.environ.setdefault("YOLO_OFFLINE", "1")
            from ultralytics import YOLO
            from common import ROOT
            self.model = YOLO(ROOT / self.weights)

    def detect(self, image) -> list[tuple[float, list[float]]]:
        if self.mock:
            return [(c, b) for c, b in self.saved[Path(image).name]]
        source = str(image) if isinstance(image, (str, Path)) else image  # 也接受影像陣列（例如夜間模擬）
        r = self.model.predict(source, imgsz=320, conf=0.01, verbose=False)[0]
        boxes = sorted(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()), key=lambda b: b[2][0])
        return [(round(c, 3), [round(v, 1) for v in b]) for k, c, b in boxes
                if r.names[int(k)] == "head" and c >= self.min_confidence]


class EventConverter:
    """head 框＋攝影機設定＋追蹤結果 → Event JSON。和 21 節 detect_events() 同一個格式。"""

    def __init__(self, config=CONFIG, site=SITE):
        cam = read_yaml(config)["camera"]
        self.detector = {"weights": cam["detector"], "min_confidence": cam["min_confidence"]}
        self.cameras = read_yaml(site)["cameras"]
        self.mismatches: list[dict] = []  # 框數和追蹤人數對不上的影格：要人看，不能默默丟掉

    def convert(self, heads, camera: str, when: str, tracks) -> list[dict]:
        if len(heads) != len(tracks):
            self.mismatches.append({"camera": camera, "time": when, "head 框": len(heads), "追蹤人數": len(tracks)})
        cam = self.cameras[camera]
        t = pd.Timestamp(when, tz="Asia/Taipei")
        return [{
            "event_id": f"{cam['site']}{camera[-1]}-{t:%Y%m%d-%H%M%S}-{i}",
            "camera_id": camera, "site": cam["site"], "zone": cam["zone"],
            "event_time": t.isoformat(), "class": "head", "confidence": conf, "bbox": box,
            "duration_sec": duration, "worker_id": worker, "detector": self.detector,
        } for i, ((conf, box), (duration, worker)) in enumerate(zip(heads, tracks))]


class ThresholdCalibrator:
    """舊版：KMeans 從資料算門檻。新版：只讀設定檔裡**已核准**的門檻，並交代它從哪來。"""

    def __init__(self, config=CONFIG):
        cfg = read_yaml(config)
        th = cfg["threshold"]
        if cfg.get("status") == "proposed" or not th.get("approved_by") or not th.get("approved_at"):
            raise PermissionError(f"{Path(config).name} 的門檻尚未核准（approved_by={th.get('approved_by')}），不能上線")
        self.config = Path(config)
        self.value = float(th["value"])
        self.provenance = {k: str(th[k]) for k in ["chosen_on", "rule", "approved_by", "approved_at"]}


class RuleEngine:
    """模型機率 → 核准門檻 → 政策規則 → 冷卻；外加 pandas 統計（舊版 analyze）。"""

    def __init__(self, calibrator: ThresholdCalibrator, rules=RULES, site=SITE, history=None):
        self.pipe = NotifyPipeline(config=calibrator.config, rules=rules, site=site, history=history)
        assert self.pipe.threshold == calibrator.value

    def process(self, events: list[dict]) -> pd.DataFrame:
        return self.pipe.run(events)

    def analyze(self, decisions: pd.DataFrame) -> dict:
        return {"事件": len(decisions), **decisions.action.value_counts().to_dict(),
                "最多通知的工地": decisions[decisions.action == "notify"].site.mode().tolist()}


def full_history() -> pd.DataFrame:
    """上線後的事件也要算進「過去 30 天違規次數」：訓練期＋上線 4 週。"""
    return pd.concat([load_events(), load_events(DATA / "events-new-weeks.csv")], ignore_index=True)


class DetectionPipeline:
    def __init__(self, config=CONFIG, mock: bool = False, history=None):
        self.calibrator = ThresholdCalibrator(config)
        self.detector = SafetyDetector(config, mock=mock)
        self.converter = EventConverter(config)
        self.engine = RuleEngine(self.calibrator, history=full_history() if history is None else history)

    def events_of(self, frames) -> list[dict]:
        """frames = [(圖檔, 攝影機, 時間, [(持續秒數, 人員)])]"""
        out = []
        for image, camera, when, tracks in frames:
            out += self.converter.convert(self.detector.detect(image), camera, when, tracks)
        return out

    def process_frames(self, frames) -> pd.DataFrame:
        return self.engine.process(self.events_of(frames))


def load_frames(path=HERE / "frames.csv", images=DATA / "scenes" / "images" / "test"):
    """frames.csv：每張圖由哪支攝影機、何時拍的、追蹤器給的持續秒數與人員（以 ; 分隔，由左到右）。"""
    frames = []
    for r in pd.read_csv(path, keep_default_na=False).itertuples():
        tracks = list(zip(map(float, filter(None, r.duration_sec.split(";"))), filter(None, r.worker_id.split(";"))))
        frames.append((images / r.image, r.camera, r.time, tracks))
    return frames


__all__ = ["SafetyDetector", "EventConverter", "ThresholdCalibrator", "RuleEngine", "DetectionPipeline",
           "load_frames", "full_history", "summary"]
