"""19｜head 類的 precision／recall 對 confidence 門檻；用 12 節的成本法選偵測門檻。

    python 19-detection-eval/evaluate.py

門檻在 val（90 張）上選；test（90 張）只在最後打開一次，報告選定門檻的成績。
「夜間」＝把同一批圖變暗、加雜訊、模糊（detlib.night），模擬資料裡沒有的條件。
"""
import numpy as np
import pandas as pd
from ultralytics import YOLO

from detlib import HEAD, HELMET, WEIGHTS, detect, images_of, iou, load_labels, sweep

APPROVED = 0.25  # config/notify.yaml 的 camera.min_confidence
GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
SHOW = [0.05, 0.10, 0.20, 0.25, 0.30, 0.50, 0.70, 0.90]

model = YOLO(WEIGHTS)
val, test = images_of("val"), images_of("test")
n_head = sum(c == HEAD for p in val for c, _ in load_labels(p))
print("== 1. val：90 張、head 標註", n_head, "個 ==")

tables, dets = {}, {}
for night in (False, True):
    name = "夜間模擬" if night else "原圖"
    dets[name] = detect(model, val, night)
    t = pd.DataFrame(sweep(dets[name], val, HEAD, GRID))
    tables[name] = t
    print(f"\n== 2. val {name}：成本 = 10×FN + 1×FP ==")
    print(t[t["門檻"].isin(SHOW)].to_string(index=False))
    best = t.loc[t["成本"].idxmin()]
    zero = t[t["成本"] == best["成本"]]["門檻"]
    span = f"{zero.min():.2f}" if zero.min() == zero.max() else f"{zero.min():.2f}–{zero.max():.2f} 都一樣"
    print(f"最低成本 {int(best['成本'])}：門檻 {span}")

# 夜間的 FP 是什麼？和 head 標註都對不上，那和 helmet 標註呢？
fp_on_helmet = fp_total = 0
for p in val:
    gts = load_labels(p)
    for c, conf, box in dets["夜間模擬"][p.stem]:
        if c == HEAD and conf >= APPROVED and max([iou(box, b) for k, b in gts if k == HEAD] or [0]) < 0.5:
            fp_total += 1
            fp_on_helmet += max([iou(box, b) for k, b in gts if k == HELMET] or [0]) >= 0.5
print(f"\n夜間、門檻 {APPROVED} 的 FP {fp_total} 個，其中框在「有戴帽子的頭」上（與 helmet 標註 IoU >= 0.5）：{fp_on_helmet} 個")

print("\n== 3. 選定門檻（val 決定），test 只開一次 ==")
rows = []
for night in (False, True):
    t = pd.DataFrame(sweep(detect(model, test, night), test, HEAD, [0.05, APPROVED]))
    t.insert(0, "test", "夜間模擬" if night else "原圖")
    rows.append(t)
print(pd.concat(rows).to_string(index=False))
