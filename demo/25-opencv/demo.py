"""25｜OpenCV：讀圖（BGR）、轉灰階、用切片裁出 ROI、縮放、畫框、存檔。"""
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
print(f"{path.name}：type {type(img).__name__}、shape {img.shape}、dtype {img.dtype}")
missing = cv2.imread(str(ROI_DIR / "frames" / "frame_999.jpg"))
print(f"檔名打錯：{missing}（不會報錯，只回傳 None）")
try:
    missing.shape
except AttributeError as e:
    print(f"下一行才出錯 → AttributeError: {e}")

print("\n== 2. BGR：同一個像素，OpenCV 和 Pillow 的順序相反 ==")
y, x = 210, 158                                       # 通道上那箱貨物（橘褐色）
b, g, r = img[y, x]
rgb = np.asarray(Image.open(path).convert("RGB"))
print(f"cv2   img[{y}, {x}] = {img[y, x].tolist()}（B, G, R）")
print(f"Pillow arr[{y}, {x}] = {rgb[y, x].tolist()}（R, G, B）")
fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"cvtColor(BGR2RGB) 之後和 Pillow 一樣嗎？ {np.array_equal(fixed, rgb)}")

print("\n== 3. cvtColor 轉灰階：白天和夜間差多少 ==")
for name in ["frame_001.jpg", "frame_003.jpg"]:
    frame = cv2.imread(str(ROI_DIR / "frames" / name))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"{name}：彩色 {frame.shape} → 灰階 {gray.shape}，平均亮度 {gray.mean():.1f}")

print("\n== 4. 切片裁出 ROI：座標寫在 roi.json ==")
roi = json.loads((ROI_DIR / "roi.json").read_text(encoding="utf-8"))
print({k: roi[k] for k in ["name", "x1", "y1", "x2", "y2", "event_type"]})
crop = img[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
print(f"img[{roi['y1']}:{roi['y2']}, {roi['x1']}:{roi['x2']}] → {crop.shape}，占畫面 {crop.shape[0] * crop.shape[1] / (img.shape[0] * img.shape[1]):.3f}")
gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"整張平均亮度 {gray_full.mean():.1f}；只看 ROI {cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean():.1f}")
view = img[0:10, 0:10]
view[:] = 0                                           # 切片不是複製：改 view 也改到原圖
print(f"把切片塗黑，原圖左上角也變了嗎？ {int(img[0, 0].sum()) == 0}")
img = cv2.imread(str(path))                           # 重新讀一次，恢復原圖
safe = img[0:10, 0:10].copy()
safe[:] = 0
print(f"先 .copy() 再塗黑，原圖左上角變了嗎？ {int(img[0, 0].sum()) == 0}")

print("\n== 5. cv2.resize：參數是 (寬, 高)，shape 是 (高, 寬) ==")
small = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
print(f"cv2.resize(img, (160, 120)) → shape {small.shape}")
wrong = cv2.resize(img, (120, 160), interpolation=cv2.INTER_AREA)
print(f"寫反成 (120, 160) → shape {wrong.shape}，寬高比 {img.shape[1] / img.shape[0]:.3f} → {wrong.shape[1] / wrong.shape[0]:.3f}")

print("\n== 6. cv2.rectangle 畫出 ROI，cv2.imwrite 存檔 ==")
marked = img.copy()
cv2.rectangle(marked, (roi["x1"], roi["y1"]), (roi["x2"] - 1, roi["y2"] - 1), (0, 0, 255), 2)
cv2.putText(marked, roi["name"], (roi["x1"], roi["y1"] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
print(f"框線上的一點 marked[{roi['y1']}, {roi['x1'] + 50}] = {marked[roi['y1'], roi['x1'] + 50].tolist()}（B, G, R：紅色）")
print(f"原圖同一點 img[{roi['y1']}, {roi['x1'] + 50}] = {img[roi['y1'], roi['x1'] + 50].tolist()}（.copy() 之後畫，原圖沒被改）")
ok = cv2.imwrite(str(OUT / "roi-marked.jpg"), marked)
print(f"imwrite out/roi-marked.jpg → {ok}")
ok = cv2.imwrite(str(OUT / "no-such-folder" / "roi.jpg"), marked)
print(f"imwrite 到不存在的資料夾 → {ok}（一樣不報錯）")
cv2.imwrite(str(OUT / "roi-rgb-wrong.jpg"), fixed)    # 把 RGB 陣列交給 imwrite：紅藍對調
back = cv2.imread(str(OUT / "roi-rgb-wrong.jpg"))
print(f"把 RGB 陣列交給 imwrite，再讀回來：貨物 {back[y, x].tolist()}，B 和 R 對調了（原本 {[int(b), int(g), int(r)]}）")
