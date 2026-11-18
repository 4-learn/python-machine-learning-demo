"""28｜torchvision resnet18.conv1：預訓練第一層學到什麼。看權重 shape → 手寫 Sobel x → 比相關係數（和隨機權重對照）→ 特徵圖。"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
np.set_printoptions(precision=2, suppress=True)

print("== 1. resnet18 的第一層：conv1 ==")
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights).eval()
conv = model.conv1
w = conv.weight.detach()                                   # (64, 3, 7, 7)
print(conv)
print(f"weight shape {tuple(w.shape)}：{w.shape[0]} 個濾波器 × {w.shape[1]} 色 × {w.shape[2]}×{w.shape[3]}")
print(f"第一層 {w.numel():,} 個數字；整個模型 {sum(p.numel() for p in model.parameters()):,} 個")

print("\n== 2. 手寫的邊緣濾波器：Sobel x ==")
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)        # 右邊減左邊：找垂直的邊
edge = np.array([[10, 10, 200]] * 3, dtype=np.float32)    # 左暗右亮
flat = np.full((3, 3), 120, dtype=np.float32)             # 平平的一塊
print("Sobel x：\n", SOBEL_X)
print(f"左暗右亮的格子 → {(edge * SOBEL_X).sum():.1f}；平平的一塊 → {(flat * SOBEL_X).sum():.1f}")

print("\n== 3. 64 個學出來的濾波器，哪個最像 Sobel x？和亂數比 ==")
sobel7 = cv2.resize(SOBEL_X, (7, 7), interpolation=cv2.INTER_LINEAR)   # Sobel x 放大成 7×7，才能和濾波器一格一格比
gray_k = w.mean(dim=1).numpy()                             # 三個顏色平均 → (64, 7, 7)
r = np.array([np.corrcoef(k.ravel(), sobel7.ravel())[0, 1] for k in gray_k])
best = int(np.argmax(np.abs(r)))
print(f"預訓練：最像的是第 {best} 個，相關係數 {r[best]:+.3f}")
torch.manual_seed(0)
w_rand = resnet18(weights=None).conv1.weight.detach()     # 同樣的形狀，數字是亂數（沒訓練過）
r_rand = np.array([np.corrcoef(k.ravel(), sobel7.ravel())[0, 1] for k in w_rand.mean(dim=1).numpy()])
print(f"沒訓練過（亂數）：最像的只有 {np.abs(r_rand).max():.3f}")
print(f"第 {best} 個（三色平均）7×7：\n", gray_k[best])

print(f"\n== 4. 特徵圖：第 {best} 個在工廠畫面上找到什麼 ==")
img = cv2.imread(str(ROOT / "story" / "data" / "factory" / "roi" / "frames" / "frame_000.jpg"))
rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
prep = weights.transforms()                                # 17 節的前處理；這裡只借它的平均、標準差
mean = torch.tensor(prep.mean)[:, None, None]
std = torch.tensor(prep.std)[:, None, None]
x = ((rgb - mean) / std)[None]                             # (1, 3, 240, 320)，不裁切，座標對得回原圖
with torch.no_grad():
    maps = conv(x)
print(f"conv1：{tuple(x.shape)} → {tuple(maps.shape)}")
col = maps[0, best].numpy().mean(axis=0)[3:-3]             # 每一行（x）平均；去掉邊上補 0 造成的假邊
hi, lo = int(np.argmax(col)) + 3, int(np.argmin(col)) + 3
print(f"最正的一行 x={hi}（原圖約 x={hi * 2}）{col[hi - 3]:+.2f}；最負的一行 x={lo}（原圖約 x={lo * 2}）{col[lo - 3]:+.2f}")
