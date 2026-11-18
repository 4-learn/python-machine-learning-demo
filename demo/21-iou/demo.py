"""21｜IoU 與偵測門檻：手寫 iou，數出 TP／FP／FN，再在 val 上掃門檻。"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # 不上網檢查更新

from pathlib import Path

import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "story" / "data" / "factory" / "scenes"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])      # 重疊區的左上角
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])      # 重疊區的右下角
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)  # 不重疊時寬或高是負的，取 0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


print("== 1. 手寫 iou：同一個 30×30 的框往右平移 ==")
truth = [100, 100, 130, 130]
for dx in [0, 5, 10, 15, 30]:
    print(f"右移 {dx:>2} 像素：IoU {iou(truth, [100 + dx, 100, 130 + dx, 130]):.3f}")
print(f"大小差一倍（30×30 和 30×60）：IoU {iou(truth, [100, 100, 130, 160]):.3f}")


def match(pred, truth, iou_min=0.5):
    """預測框依信心值由高到低，各自找 IoU 最大、還沒被配過的標註框。回傳每個預測框的結果與 FN 數。"""
    pred = sorted(pred, key=lambda p: -p[0])
    used, rows = set(), []
    for score, box in pred:
        ious = [iou(box, t) for t in truth]
        best, best_j = 0.0, None
        for j, v in enumerate(ious):
            if j not in used and v > best:
                best, best_j = v, j
        if best >= iou_min:
            used.add(best_j)          # TP；沒配上的就是 FP
            rows.append({"conf": score, "最重疊": f"標註{best_j}", "IoU": round(best, 3), "結果": "TP"})
        else:
            top = max(ious, default=0.0)
            why = "FP（重複框）" if top >= iou_min else "FP（位置不對）"
            rows.append({"conf": score, "最重疊": f"標註{ious.index(top)}" if top > 0 else "-",
                         "IoU": round(top, 3), "結果": why})
    return rows, len(truth) - len(used)


print("\n== 2. 對齊規則：IoU >= 0.5，一個標註框只配一次 ==")
truth = [[100.0, 100.0, 130.0, 130.0], [200.0, 100.0, 230.0, 130.0], [300.0, 100.0, 330.0, 130.0]]
pred = [(0.95, [101.0, 101.0, 131.0, 131.0]),   # 對準標註0
        (0.60, [103.0, 99.0, 133.0, 129.0]),    # 又框了一次標註0
        (0.90, [215.0, 100.0, 245.0, 130.0]),   # 偏了半個框
        (0.40, [50.0, 50.0, 70.0, 70.0])]       # 框在沒人的地方
rows, fn = match(pred, truth)
print(pd.DataFrame(rows).to_string(index=False))
tp = sum(r["結果"] == "TP" for r in rows)
print(f"TP {tp}、FP {len(rows) - tp}、FN {fn}（標註1、標註2 沒被配到）")
print(f"precision {tp / len(rows):.3f}、recall {tp / len(truth):.3f}")

print("\n== 3. 工廠線 val 60 張：先用 conf=0.01 全部收下，再掃門檻 ==")
model = YOLO(str(ROOT / "models" / "factory-yolo11n.pt"))
pred_by_image, truth_by_image = {}, {}
for r in model.predict([str(p) for p in sorted((SCENES / "images" / "val").glob("*.jpg"))],
                       conf=0.01, imgsz=320, device="cpu", verbose=False, stream=True):
    name = Path(r.path).stem
    pred_by_image[name] = [(round(s, 3), b) for c, s, b in
                           zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()) if c == 1]
    truth_by_image[name] = []
    for line in (SCENES / "labels" / "val" / f"{name}.txt").read_text().splitlines():
        cls, cx, cy, w, h = map(float, line.split())
        if cls == 1:
            truth_by_image[name].append([(cx - w / 2) * 320, (cy - h / 2) * 320, (cx + w / 2) * 320, (cy + h / 2) * 320])
print(f"head 標註 {sum(map(len, truth_by_image.values()))} 個；"
      f"head 預測框 {sum(map(len, pred_by_image.values()))} 個（conf >= 0.01）")
sweep = []
for conf in [0.95, 0.9, 0.5, 0.05, 0.02, 0.01]:
    tp = fp = fn = 0
    for name in truth_by_image:
        rows, missed = match([p for p in pred_by_image[name] if p[0] >= conf], truth_by_image[name])
        tp += sum(r["結果"] == "TP" for r in rows)
        fp += sum(r["結果"] != "TP" for r in rows)
        fn += missed
    sweep.append({"conf": conf, "TP": tp, "FP": fp, "FN": fn,
                  "precision": round(tp / (tp + fp), 3), "recall": round(tp / (tp + fn), 3)})
print(pd.DataFrame(sweep).to_string(index=False))
