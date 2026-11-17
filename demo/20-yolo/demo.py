"""20｜ultralytics YOLO：載入教師訓練好的偵測器，對場景照片推論，拆開每一個偵測框。"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # 不上網檢查更新

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "story" / "data" / "factory" / "scenes"

print("== 1. 載入權重檔：認得哪些類別、怎麼訓練的 ==")
model = YOLO(str(ROOT / "models" / "factory-yolo11n.pt"))
args = model.ckpt["train_args"]
print(f"類別：{model.names}")
print(f"起點 {Path(args['model']).name}，{args['epochs']} epochs，imgsz {args['imgsz']}，device {args['device']}")
print(f"授權：{model.ckpt['license']}")
coco = YOLO(str(ROOT / "models" / "yolo11n.pt"))
print(f"COCO 預訓練的 yolo11n.pt：{len(coco.names)} 類，前 3 類 {[coco.names[i] for i in range(3)]}，"
      f"有 helmet 嗎：{'helmet' in coco.names.values()}")

print("\n== 2. predict：一張圖得到一個 Results ==")
image = SCENES / "images" / "test" / "test_001.jpg"
results = model.predict(str(image), conf=0.25, imgsz=320, device="cpu", verbose=False)
r = results[0]
print(f"回傳 {type(results).__name__}，長度 {len(results)}；裡面是 {type(r).__name__}")
print(f"原圖大小（高, 寬）：{r.orig_shape}；框數：{len(r.boxes)}")
print(f"boxes.xyxy 形狀：{tuple(r.boxes.xyxy.shape)}（每列 x1, y1, x2, y2，單位像素）")
print(f"boxes.conf：{[round(c, 3) for c in r.boxes.conf.tolist()]}")
print(f"boxes.cls ：{r.boxes.cls.tolist()}")

print("\n== 3. 一個框一列：類別名稱、信心值、位置 ==")
print(" class   conf     x1     y1     x2     y2")
for cls, score, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
    print(f"{r.names[int(cls)]:>6}  {score:.3f}  " + "  ".join(f"{v:5.1f}" for v in box))

print("\n== 4. 標註檔（YOLO 格式：類別 中心x 中心y 寬 高，0–1 的比例）==")
label = SCENES / "labels" / "test" / "test_001.txt"
print(label.read_text().strip())
cls, cx, cy, w, h = map(float, label.read_text().splitlines()[0].split())
x1, y1, x2, y2 = (cx - w / 2) * 320, (cy - h / 2) * 320, (cx + w / 2) * 320, (cy + h / 2) * 320
print(f"第一列換回像素：x1 {x1:.1f}  y1 {y1:.1f}  x2 {x2:.1f}  y2 {y2:.1f}")

print("\n== 5. conf：信心值低於門檻的框不回傳（test_007）==")
other = SCENES / "images" / "test" / "test_007.jpg"
for conf in [0.01, 0.25]:
    b = model.predict(str(other), conf=conf, imgsz=320, device="cpu", verbose=False)[0].boxes
    pairs = [f"{model.names[int(c)]} {s:.3f}" for c, s in zip(b.cls.tolist(), b.conf.tolist())]
    print(f"conf={conf:<4}  框數 {len(b)}：{'、'.join(pairs)}")

print("\n== 6. stream=True：很多張圖時，一次給一張 ==")
images = [str(p) for p in sorted((SCENES / "images" / "test").glob("*.jpg"))[:5]]
gen = model.predict(images, conf=0.25, imgsz=320, device="cpu", verbose=False, stream=True)
print(f"回傳的是 {type(gen).__name__}，還沒開始算")
for res in gen:
    names = [res.names[int(c)] for c in res.boxes.cls.tolist()]
    print(f"{Path(res.path).name}：helmet {names.count('helmet')} 個、head {names.count('head')} 個")
