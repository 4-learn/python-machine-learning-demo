"""19｜IoU 與 TP／FP／FN：幾張圖手算一次。

    python 19-detection-eval/iou.py
"""
import pandas as pd
from ultralytics import YOLO

from detlib import HEAD, HELMET, WEIGHTS, detect, images_of, iou, load_labels, match

NAMES = {HELMET: "helmet", HEAD: "head"}

print("== 1. IoU 的直覺：同一個 30×30 的框往右平移 ==")
a = [100, 100, 130, 130]
for dx in (0, 5, 10, 15, 20, 30):
    print(f"右移 {dx:2d} 像素：IoU {iou(a, [100 + dx, 100, 130 + dx, 130]):.3f}")
print(f"大小差一倍（30×30 vs 30×60）：IoU {iou(a, [100, 100, 130, 160]):.3f}")

model = YOLO(WEIGHTS)


def show(image, night_mode=False):
    dets = detect(model, [image], night_mode=night_mode)[image.stem]
    gts = load_labels(image)
    rows = []
    for c, conf, box in sorted(dets, key=lambda d: -d[1]):
        if conf < 0.25:
            continue
        best = max((iou(box, b), j) for j, (_, b) in enumerate(gts))
        rows.append({"預測": NAMES[c], "conf": round(conf, 3), "最重疊的標註": f"#{best[1]} {NAMES[gts[best[1]][0]]}",
                     "IoU": round(best[0], 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("標註：", ", ".join(f"#{j} {NAMES[k]}" for j, (k, _) in enumerate(gts)))
    tp, fp, fn = match(dets, gts, HEAD, 0.25)
    print(f"head 類（門檻 0.25、IoU >= 0.5）：TP {tp}  FP {fp}  FN {fn}")


test = images_of("test")
print("\n== 2. test_014 原圖：預測框 vs 標註框 ==")
show(test[14])
print("\n== 3. test_052 夜間模擬 ==")
show(test[52], night_mode=True)
print("\n== 4. test_058 夜間模擬 ==")
show(test[58], night_mode=True)
