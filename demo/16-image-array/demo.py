"""16｜Pillow 與 NumPy：一張照片在程式裡就是一個 (高, 寬, 3) 的 uint8 陣列。"""
import numpy as np
from PIL import Image

PHOTO = "story/data/photos/photo-site-rebar.jpg"              # AI 生成的擬真工地照片，不是真實人物
CROPS = "story/data/factory/crops"

print("== 1. 一張照片＝一個 3 維陣列 ==")
img = Image.open(PHOTO)
arr = np.asarray(img)
print(f"Pillow：{img.width}×{img.height}（寬×高），模式 {img.mode}")
print(f"NumPy ：shape {arr.shape}（高, 寬, 顏色），dtype {arr.dtype}，最小 {arr.min()}，最大 {arr.max()}")
print(f"總共 {arr.size:,} 個數字")

print("\n== 2. 一個像素＝三個 0–255 的整數（R, G, B）==")
POINTS = [("黃色安全帽", 362, 268), ("白色安全帽", 698, 318),   # (說明, x, y)：教師在照片上量的位置
          ("沒戴帽子的頭髮", 560, 262), ("水泥地面", 470, 535)]
for name, x, y in POINTS:
    r, g, b = arr[y, x]
    print(f"{name:<8} arr[{y}, {x}] = ({r:3d}, {g:3d}, {b:3d})")

print("\n== 3. 先 y 後 x：切片就是裁切 ==")
x1, y1, x2, y2 = 315, 240, 410, 320
patch = arr[y1:y2, x1:x2]
print(f"arr[{y1}:{y2}, {x1}:{x2}] → shape {patch.shape}")
print(f"和 img.crop(({x1}, {y1}, {x2}, {y2})) 一樣嗎？",
      np.array_equal(patch, np.asarray(img.crop((x1, y1, x2, y2)))))

print("\n== 4. 灰階與亮度：白天和夜間的小圖 ==")
for name in ["factory/crops/ok/camera_01_001.jpg", "factory/crops/ok/camera_04_002.jpg",
             "care/crops/ok/camera_11_000.jpg", "care/crops/ok/camera_14_002.jpg"]:
    small = Image.open(f"story/data/{name}")
    gray = np.asarray(small.convert("L"))
    print(f"{name:<35} 彩色 {np.asarray(small).shape} → 灰階 {gray.shape}，"
          f"平均亮度 {gray.mean():6.1f}，暗像素（<50）占 {(gray < 50).mean():.3f}")

print("\n== 5. uint8 的陷阱：250 + 10 不是 260 ==")
p = np.array([250, 128, 10], dtype=np.uint8)
print("uint8   + 10 →", (p + 10).tolist())
print("int16   + 10 →", (p.astype(np.int16) + 10).tolist())
print("float32 / 255 →", [round(float(v), 3) for v in p.astype(np.float32) / 255])

print("\n== 6. 縮放：resize 寫 (寬, 高)，shape 是 (高, 寬) ==")
wide = img.resize((224, 112))
print(f"img.resize((224, 112)) → {wide.width}×{wide.height}，陣列 shape {np.asarray(wide).shape}")
square = img.resize((224, 224))
print(f"直接壓成 224×224：寬高比 {img.width / img.height:.2f} → {square.width / square.height:.2f}（被壓扁）")
scale = 256 / min(img.size)
short = img.resize((round(img.width * scale), round(img.height * scale)))
left, top = (short.width - 224) // 2, (short.height - 224) // 2
center = short.crop((left, top, left + 224, top + 224))
print(f"短邊縮到 256 → {short.width}×{short.height}，再從中間裁 → {center.width}×{center.height}"
      f"（左右共裁掉 {short.width - 224} 像素、上下共 {short.height - 224} 像素）")

print("\n== 7. 走到模型要的樣子：(1, 3, 224, 224) 的 float32 ==")
MEAN = np.array([0.485, 0.456, 0.406])          # ImageNet 照片每個顏色的平均
STD = np.array([0.229, 0.224, 0.225])           # 和標準差
x01 = np.asarray(center).astype(np.float32) / 255
xn = ((x01 - MEAN) / STD).astype(np.float32)
chw = xn.transpose(2, 0, 1)
one = chw[np.newaxis]
print(f"/255 後範圍 {x01.min():.3f}～{x01.max():.3f}；標準化後 {xn.min():.3f}～{xn.max():.3f}")
print(f"transpose(2, 0, 1)：{xn.shape} → {chw.shape}；前面加一維 → {one.shape}，dtype {one.dtype}")
wrong = xn.reshape(3, 224, 224)
print(f"reshape(3, 224, 224) 的 shape 也是 {wrong.shape}，內容和 transpose 一樣嗎？",
      np.array_equal(wrong, chw))
crops = [np.asarray(Image.open(f"{CROPS}/{n}").convert("RGB").resize((224, 224)))
         for n in ["ok/camera_01_001.jpg", "alert/camera_01_000.jpg", "ok/camera_04_002.jpg"]]
batch = np.stack(crops)
print(f"3 張小圖 np.stack → {batch.shape}；transpose(0, 3, 1, 2) → {batch.transpose(0, 3, 1, 2).shape}")
