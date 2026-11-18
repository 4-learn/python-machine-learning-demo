"""25｜OpenCV：讀圖（BGR）、用切片裁出 ROI、畫框存檔。"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ROI_DIR = ROOT / "story" / "data" / "factory" / "roi"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

print("== 1. cv2.imread：讀進來就是 NumPy 陣列 ==")
path = ROI_DIR / "frames" / "frame_001.jpg"          # 白天、通道上有一箱貨物、有人走過
img = cv2.imread(str(path))
print(f"{path.name}：shape {img.shape}、dtype {img.dtype}")
missing = cv2.imread(str(ROI_DIR / "frames" / "frame_999.jpg"))
print(f"檔名打錯：{missing}（不會報錯，只回傳 None）")

print("\n== 2. BGR：同一個像素，OpenCV 和 Pillow 的順序相反 ==")
y, x = 210, 158                                       # 通道上那箱貨物（橘褐色）
rgb = np.asarray(Image.open(path).convert("RGB"))
print(f"cv2   img[{y}, {x}] = {img[y, x].tolist()}（B, G, R）")
print(f"Pillow arr[{y}, {x}] = {rgb[y, x].tolist()}（R, G, B）")
fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"cvtColor(BGR2RGB) 之後和 Pillow 一樣嗎？ {np.array_equal(fixed, rgb)}")

print("\n== 3. 切片裁出 ROI：座標寫在 roi.json ==")
roi = json.loads((ROI_DIR / "roi.json").read_text(encoding="utf-8"))
print({k: roi[k] for k in ["name", "x1", "y1", "x2", "y2", "event_type"]})
crop = img[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
print(f"img[{roi['y1']}:{roi['y2']}, {roi['x1']}:{roi['x2']}] → {crop.shape}")
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
print(f"ROI 轉灰階 → {gray.shape}，平均亮度 {gray.mean():.1f}")

print("\n== 4. cv2.rectangle 畫出 ROI，cv2.imwrite 存檔 ==")
marked = img.copy()
cv2.rectangle(marked, (roi["x1"], roi["y1"]), (roi["x2"] - 1, roi["y2"] - 1), (0, 0, 255), 2)
cv2.putText(marked, roi["name"], (roi["x1"], roi["y1"] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
ok = cv2.imwrite(str(OUT / "roi-marked.jpg"), marked)
print(f"框線上的一點 marked[{roi['y1']}, {roi['x1'] + 50}] = {marked[roi['y1'], roi['x1'] + 50].tolist()}（B, G, R：紅色）")
print(f"imwrite out/roi-marked.jpg → {ok}")
