"""10｜先用官方模型：COCO 有的東西找得到、沒有的東西找不到。

在 repo 根目錄執行：
    python demo/20-yolo/coco_vs_helmet.py
"""
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import ASSETS

ROOT = Path(__file__).resolve().parents[2]

# 只寫檔名 YOLO("yolo11n.pt") 會從網路下載；這裡用 repo 已附的那一份，不用網路
model = YOLO(str(ROOT / "models" / "yolo11n.pt"))
print(f"== 官方模型認得 {len(model.names)} 類：{model.names[0]}、{model.names[1]}、{model.names[2]}……")
print("有沒有 helmet：", any("helmet" in n for n in model.names.values()))

photos = [("COCO 有的東西（公車、人）", ASSETS / "bus.jpg"),
          ("COCO 沒有的東西（安全帽）", ROOT / "story" / "data" / "photos" / "ppe_site.jpg")]
for title, path in photos:
    r = model.predict(str(path), conf=0.25, device="cpu", verbose=False)[0]
    print(f"\n== {title}：{path.name}，找到 {len(r.boxes)} 個框 ==")
    for cls, score in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        print(f"  {model.names[int(cls)]:<8} {score:.2f}")
