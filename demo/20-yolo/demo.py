"""20｜ultralytics YOLO：載入教師訓練好的偵測器，對場景照片推論，拆開每一個偵測框。"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # 不上網檢查更新

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "story" / "data" / "factory" / "scenes"

print("== 1. 載入偵測器：認得哪些類別、從哪裡來 ==")
model = YOLO(str(ROOT / "models" / "factory-yolo11n.pt"))
args = model.ckpt["train_args"]
print(f"類別：{model.names}")
print(f"起點 {Path(args['model']).name}，{args['epochs']} 輪，圖片大小 {args['imgsz']}")
print(f"授權：{model.ckpt['license']}")

print("\n== 2. predict：一個框一列 ==")
image = SCENES / "images" / "test" / "test_001.jpg"
results = model.predict(str(image), conf=0.25, imgsz=320, device="cpu", verbose=False)
r = results[0]
print(f"框數：{len(r.boxes)}")
print(" class   conf     x1     y1     x2     y2")
for cls, score, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
    print(f"{r.names[int(cls)]:>6}  {score:.3f}  " + "  ".join(f"{v:5.1f}" for v in box))

print("\n== 3. 標註檔（YOLO 格式：類別 中心x 中心y 寬 高，0–1 的比例）==")
label = SCENES / "labels" / "test" / "test_001.txt"
print(label.read_text().strip())
cls, cx, cy, w, h = map(float, label.read_text().splitlines()[0].split())
x1, y1, x2, y2 = (cx - w / 2) * 320, (cy - h / 2) * 320, (cx + w / 2) * 320, (cy + h / 2) * 320
print(f"第一列換回像素：x1 {x1:.1f}  y1 {y1:.1f}  x2 {x2:.1f}  y2 {y2:.1f}")
