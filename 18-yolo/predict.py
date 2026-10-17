"""18｜YOLO 推論：教師訓練的安全帽偵測器 vs COCO 預訓練模型。

    YOLO_OFFLINE=1 python 18-yolo/predict.py
"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # 不連網檢查更新

import pandas as pd
from ultralytics import YOLO

from common import DATA, ROOT

HELMET = ROOT / "models" / "helmet-yolo11n.pt"
COCO = ROOT / "models" / "yolo11n.pt"
PHOTOS = ["photo-site-rebar.jpg", "photo-warehouse-night.jpg", "photo-lunch-break.jpg"]


def boxes_table(result):
    """把一張圖的 Results 變成 DataFrame：一列一個框。"""
    b = result.boxes
    df = pd.DataFrame(b.xyxy.tolist(), columns=["x1", "y1", "x2", "y2"]).round(1)
    df.insert(0, "class", [result.names[int(c)] for c in b.cls.tolist()])
    df.insert(1, "conf", [round(c, 3) for c in b.conf.tolist()])
    return df


helmet = YOLO(HELMET)
coco = YOLO(COCO)

print("== 1. 兩個權重檔 ==")
for name, m in [("helmet-yolo11n.pt", helmet), ("yolo11n.pt", coco)]:
    args = m.ckpt["train_args"]
    print(f"{name}: 類別 {len(m.names)} 個，訓練 {args['epochs']} epochs，imgsz {args['imgsz']}，授權 {m.ckpt['license'].split()[0]}")
print("helmet 類別：", helmet.names)
print("COCO 前 5 類：", {k: coco.names[k] for k in range(5)}, "…有沒有 helmet：", "helmet" in coco.names.values())

print("\n== 2. 一張合成場景：Results 物件 ==")
scene = DATA / "scenes" / "images" / "test" / "test_000.jpg"
r = helmet.predict(scene, verbose=False)[0]
print("type:", type(r).__name__, "| 原圖大小 (高, 寬):", r.orig_shape, "| 框數:", len(r.boxes))
print("boxes.xyxy 形狀:", tuple(r.boxes.xyxy.shape), "（每列 x1, y1, x2, y2，單位像素）")
print("boxes.conf:", [round(c, 3) for c in r.boxes.conf.tolist()])
print("boxes.cls :", r.boxes.cls.tolist())
print(boxes_table(r).to_string(index=False))
print("標註檔（YOLO 格式：類別 中心x 中心y 寬 高，比例）：")
print((DATA / "scenes" / "labels" / "test" / "test_000.txt").read_text().strip())

print("\n== 3. 擬真照片：imgsz 320 vs 640 ==")
rows = []
for photo in PHOTOS:
    for size in (320, 640):
        res = helmet.predict(DATA / photo, imgsz=size, verbose=False)[0]
        t = boxes_table(res)
        rows.append({"photo": photo.removeprefix("photo-").removesuffix(".jpg"), "imgsz": size,
                     "head": " ".join(f"{c:.2f}" for c in t[t["class"] == "head"].conf),
                     "helmet": " ".join(f"{c:.2f}" for c in t[t["class"] == "helmet"].conf)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 4. 同三張照片，COCO 預訓練 yolo11n ==")
for photo in PHOTOS:
    t = boxes_table(coco.predict(DATA / photo, verbose=False)[0])
    print(photo.removeprefix("photo-").removesuffix(".jpg"), "→", ", ".join(f"{c} {v:.2f}" for c, v in zip(t["class"], t.conf)))
t = boxes_table(coco.predict(scene, verbose=False)[0])
print("合成場景 test_000 →", len(t), "個框")
