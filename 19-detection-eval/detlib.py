"""19｜偵測評估的小工具：讀 YOLO 標註、IoU、逐框配對、夜間模擬。

這裡的函式刻意寫得短，讓你看得到每一步；21 節以後可以 import 它。
"""
from __future__ import annotations

import os
import zlib

os.environ.setdefault("YOLO_OFFLINE", "1")

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from common import DATA, ROOT

SCENES = DATA / "scenes"
WEIGHTS = ROOT / "models" / "helmet-yolo11n.pt"
HELMET, HEAD = 0, 1
FN_COST, FP_COST = 10, 1  # 12 節：漏報一次＝誤報十次


def iou(a, b) -> float:
    """兩個 xyxy 框的 IoU＝交集面積 / 聯集面積。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def load_labels(image: Path, size: int = 320) -> list[tuple[int, list[float]]]:
    """YOLO 標註（類別 中心x 中心y 寬 高，0–1 比例）→ [(類別, 像素 xyxy)]。"""
    txt = SCENES / "labels" / image.parent.name / (image.stem + ".txt")
    out = []
    for line in txt.read_text().splitlines():
        if line.strip():
            c, x, y, w, h = map(float, line.split())
            out.append((int(c), [(x - w / 2) * size, (y - h / 2) * size, (x + w / 2) * size, (y + h / 2) * size]))
    return out


def night(image: Path) -> np.ndarray:
    """夜間模擬：變暗、加雜訊、變模糊。雜訊種子由檔名決定，同一張圖每次都一樣。
    回傳 BGR 陣列（ultralytics 對陣列採 BGR）。"""
    rng = np.random.default_rng(zlib.crc32(f"{image.parent.name}/{image.name}".encode()))
    a = np.asarray(Image.open(image).convert("RGB"), dtype=np.float32) * 0.35 + rng.normal(0, 14, (320, 320, 3))
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    return np.asarray(img)[:, :, ::-1].copy()


def detect(model, images: list[Path], night_mode: bool = False, min_conf: float = 0.01):
    """對每張圖推論，回傳 {圖名: [(類別, conf, xyxy), ...]}。門檻先放很低，之後再掃。"""
    sources = [night(p) for p in images] if night_mode else [str(p) for p in images]
    out = {}
    for p, r in zip(images, model.predict(sources, conf=min_conf, imgsz=320, verbose=False, stream=True)):
        out[p.stem] = list(zip([int(c) for c in r.boxes.cls.tolist()], r.boxes.conf.tolist(), r.boxes.xyxy.tolist()))
    return out


def match(dets, gts, cls: int, threshold: float, iou_min: float = 0.5):
    """單張圖、單一類別：信心由高到低，每個預測框找 IoU 最大且還沒被配過的標註框。"""
    preds = sorted([d for d in dets if d[0] == cls and d[1] >= threshold], key=lambda d: -d[1])
    truth = [b for c, b in gts if c == cls]
    used, tp = set(), 0
    for _, _, box in preds:
        best, best_j = 0.0, None
        for j, t in enumerate(truth):
            if j not in used and iou(box, t) > best:
                best, best_j = iou(box, t), j
        if best >= iou_min:
            tp += 1
            used.add(best_j)
    return tp, len(preds) - tp, len(truth) - tp  # TP, FP, FN


def sweep(all_dets, images, cls: int, thresholds):
    rows = []
    for t in thresholds:
        tp = fp = fn = 0
        for p in images:
            a, b, c = match(all_dets[p.stem], load_labels(p), cls, t)
            tp, fp, fn = tp + a, fp + b, fn + c
        rows.append({"門檻": t, "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
                     "recall": round(tp / (tp + fn), 3),
                     "成本": FN_COST * fn + FP_COST * fp})
    return rows


def images_of(split: str) -> list[Path]:
    return sorted((SCENES / "images" / split).glob("*.jpg"))
