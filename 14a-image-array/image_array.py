"""D0｜影像就是陣列：像素、shape、裁切、亮度、uint8、BGR、縮放，最後走到模型要的 (1, 3, 224, 224)。"""
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights

from common import DATA

SITE = DATA / "photo-site-rebar.jpg"

print("== 1. 一張照片＝一個 3 維陣列 ==")
img = Image.open(SITE)
arr = np.asarray(img)
print(f"PIL：{img.width}×{img.height}（寬×高），模式 {img.mode}")
print(f"陣列：shape {arr.shape}（高, 寬, 顏色），dtype {arr.dtype}，最小 {arr.min()}，最大 {arr.max()}")
print(f"總共 {arr.size:,} 個數字")

print("\n== 2. 一個像素＝三個 0–255 的整數（R, G, B）==")
POINTS = [  # (說明, x, y)：教師在照片上量的位置
    ("黃色安全帽", 362, 268),
    ("白色安全帽", 698, 318),
    ("沒戴帽子的頭髮", 560, 262),
    ("水泥地面", 470, 535),
]
for name, x, y in POINTS:
    r, g, b = arr[y, x]
    print(f"{name:<10} arr[{y}, {x}] = ({r:3d}, {g:3d}, {b:3d})")

print("\n== 3. 先 y 後 x：切片就是裁切 ==")
x1, y1, x2, y2 = 315, 240, 410, 320
patch = arr[y1:y2, x1:x2]
same = np.array_equal(patch, np.asarray(img.crop((x1, y1, x2, y2))))
print(f"arr[{y1}:{y2}, {x1}:{x2}] → shape {patch.shape}")
print(f"和 PIL 的 img.crop(({x1}, {y1}, {x2}, {y2})) 一樣嗎？ {same}")

print("\n== 4. 亮度：把「看起來比較暗」變成數字 ==")
for path in sorted(DATA.glob("photo-*.jpg")):
    gray = np.asarray(Image.open(path).convert("L"))
    print(f"{path.name:<26} 灰階 shape {gray.shape}，平均亮度 {gray.mean():6.1f}，"
          f"暗像素（<50）占 {(gray < 50).mean():.1%}")

print("\n== 5. uint8 的陷阱：250 + 10 不是 260 ==")
p = np.array([250, 128, 10], dtype=np.uint8)
print("uint8   + 10 →", (p + 10).tolist())
print("int16   + 10 →", (p.astype(np.int16) + 10).tolist())
print("float32 / 255 →", (p.astype(np.float32) / 255).round(3).tolist())

print("\n== 6. OpenCV 讀進來是 BGR ==")
bgr = cv2.imread(str(SITE))
x, y = POINTS[0][1], POINTS[0][2]
print(f"PIL    arr[{y}, {x}] = {tuple(int(v) for v in arr[y, x])}  (R, G, B)")
print(f"OpenCV bgr[{y}, {x}] = {tuple(int(v) for v in bgr[y, x])}  (B, G, R)")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
print("cv2.cvtColor 轉成 RGB 後和 PIL 一樣？", np.array_equal(rgb, arr))
print("cv2.imread 讀不存在的檔案 →", cv2.imread("no-such-file.jpg"))

print("\n== 7. 縮放：尺寸寫 (寬, 高)，shape 是 (高, 寬) ==")
small = img.resize((224, 224))
print(f"img.resize((224, 224)) → shape {np.asarray(small).shape}；原本寬高比 {img.width / img.height:.2f}，變成 1.00（被壓扁）")
keep = img.resize((256, round(256 * img.height / img.width)))
print(f"等比例把寬縮到 256 → {keep.width}×{keep.height}")
side = 256 / min(img.size)
short = img.resize((round(img.width * side), round(img.height * side)))
left, top = (short.width - 224) // 2, (short.height - 224) // 2
center = short.crop((left, top, left + 224, top + 224))
print(f"短邊縮到 256 → {short.width}×{short.height}，再從中間裁 224×224 → {center.width}×{center.height}")

print("\n== 8. 走到模型要的樣子：(1, 3, 224, 224) 的 float32 ==")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
x01 = np.asarray(center).astype(np.float32) / 255           # 0–255 → 0–1
xn = (x01 - MEAN) / STD                                      # 每個顏色減平均、除標準差
chw = xn.transpose(2, 0, 1)                                  # (224,224,3) → (3,224,224)
batch = torch.from_numpy(chw).float().unsqueeze(0)           # → (1,3,224,224)
print(f"/255 後範圍 {x01.min():.2f}～{x01.max():.2f}；標準化後 {xn.min():.2f}～{xn.max():.2f}")
print(f"transpose 後 {chw.shape}；unsqueeze 後 {tuple(batch.shape)}，dtype {batch.dtype}")
official = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()(img.convert("RGB")).unsqueeze(0)
diff = (batch - official).abs()
print(f"和 weights.transforms() 比：shape {tuple(official.shape)}，平均差 {diff.mean():.3f}，最大差 {diff.max():.3f}")
