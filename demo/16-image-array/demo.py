"""16｜Pillow 與 NumPy：一張照片在程式裡就是一個 (高, 寬, 3) 的 uint8 陣列。"""
import numpy as np
from PIL import Image

CROPS = "story/data/factory/crops"

print("== 1. 一張照片＝一個 3 維陣列 ==")
# photo-site-rebar.jpg 是 AI 生成的擬真工地照片，不是真實人物
img = Image.open("story/data/photos/photo-site-rebar.jpg")
arr = np.asarray(img)
print(img.width, img.height, img.mode)
print(arr.shape, arr.dtype, arr.min(), arr.max())
# (說明, x, y)：教師在照片上量的位置
POINTS = [("黃色安全帽", 362, 268), ("白色安全帽", 698, 318),
          ("沒戴帽子的頭髮", 560, 262), ("水泥地面", 470, 535)]
for name, x, y in POINTS:
    r, g, b = arr[y, x]
    print(f"{name:<8} arr[{y}, {x}] = ({r:3d}, {g:3d}, {b:3d})")

print("\n== 2. 先 y 後 x：切片就是裁切 ==")
x1, y1, x2, y2 = 315, 240, 410, 320
patch = arr[y1:y2, x1:x2]
print(f"arr[{y1}:{y2}, {x1}:{x2}] → shape {patch.shape}")
print(f"和 img.crop(({x1}, {y1}, {x2}, {y2})) 一樣嗎？",
      np.array_equal(patch, np.asarray(img.crop((x1, y1, x2, y2)))))

print("\n== 3. 灰階：把「比較暗」變成數字 ==")
for name in ["factory/crops/ok/camera_01_001.jpg", "factory/crops/ok/camera_04_002.jpg",
             "care/crops/ok/camera_11_000.jpg", "care/crops/ok/camera_14_002.jpg"]:
    gray = np.asarray(Image.open(f"story/data/{name}").convert("L"))
    print(f"{name:<35} 灰階 {gray.shape}，平均亮度 {gray.mean():6.1f}")

print("\n== 4. 變成模型要的 (1, 3, 224, 224) ==")
p = np.array([250, 128, 10], dtype=np.uint8)
print("uint8 直接 + 10：", p + 10)
print("先轉 float32 再 / 255：", p.astype(np.float32) / 255)
# ImageNet 照片每個顏色的平均和標準差
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
small = Image.open(f"{CROPS}/ok/camera_01_001.jpg").convert("RGB").resize((224, 224))
x01 = np.asarray(small).astype(np.float32) / 255
xn = ((x01 - MEAN) / STD).astype(np.float32)
chw = xn.transpose(2, 0, 1)
one = chw[np.newaxis]
print(f"縮成 224×224 {x01.shape}；/255 後 {x01.min():.3f}～{x01.max():.3f}；標準化後 {xn.min():.3f}～{xn.max():.3f}")
print(f"transpose(2, 0, 1) → {chw.shape}；前面加一維 → {one.shape}，dtype {one.dtype}")
crops = [np.asarray(Image.open(f"{CROPS}/{n}").convert("RGB").resize((224, 224)))
         for n in ["ok/camera_01_001.jpg", "alert/camera_01_000.jpg", "ok/camera_04_002.jpg"]]
batch = np.stack(crops).transpose(0, 3, 1, 2)
print(f"3 張小圖 np.stack 再 transpose(0, 3, 1, 2) → {batch.shape}")
