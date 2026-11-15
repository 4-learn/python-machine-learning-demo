"""28｜PyTorch nn.Conv2d：卷積層就是一疊濾波器。手算一個位置 → cv2.filter2D → 把 Sobel 放進 nn.Conv2d → 參數量 → 預訓練模型第一層。"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18

torch.manual_seed(0)
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
np.set_printoptions(precision=3, suppress=True)

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)       # 左暗右亮的地方是正的：找垂直的邊

print("== 1. 手算一個位置：3×3 的格子 × 核，乘完加起來 ==")
patch = np.array([[10, 10, 200],
                  [10, 10, 200],
                  [10, 10, 200]], dtype=np.float32)      # 左邊暗、右邊亮：一條垂直的邊
flat = np.full((3, 3), 120, dtype=np.float32)            # 平平的一塊
print("格子（左暗右亮）：\n", patch)
print("核（Sobel x）：\n", SOBEL_X)
print(f"(格子 * 核).sum() = {(patch * SOBEL_X).sum():.1f}")
print(f"平平的一塊 = {(flat * SOBEL_X).sum():.1f}；核的總和 = {SOBEL_X.sum():.1f}")

print("\n== 2. cv2.filter2D：核滑過整張圖，每個位置都算一次 ==")
img = cv2.imread(str(ROOT / "story" / "data" / "factory" / "roi" / "frames" / "frame_000.jpg"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
edges_cv = cv2.filter2D(gray, -1, SOBEL_X, borderType=cv2.BORDER_CONSTANT)
y, x = 200, 109                                            # 通道左邊黃線的左緣：左邊地板暗、右邊黃線亮
by_hand = (gray[y - 1:y + 2, x - 1:x + 2] * SOBEL_X).sum()
print(f"輸入 {gray.shape} → 輸出 {edges_cv.shape}")
print(f"位置 (y={y}, x={x}) 的 3×3：\n", gray[y - 1:y + 2, x - 1:x + 2])
print(f"手算 {by_hand:.3f}、filter2D {edges_cv[y, x]:.3f}；旁邊平平的地板 (y=220, x=90)：{edges_cv[220, 90]:.3f}")
col = np.abs(edges_cv[:, 2:-2]).mean(axis=0)
print("每一行（x）平均邊緣強度最大的 4 個 x：", sorted((np.argsort(col)[-4:] + 2).tolist()))

print("\n== 3. nn.Conv2d：同樣的核放進 PyTorch ==")
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
print(conv)
print(f"weight shape {tuple(conv.weight.shape)}（輸出通道, 輸入通道, 核高, 核寬）")
print("剛建好時的核（隨機）：\n", conv.weight[0, 0].detach().numpy())
with torch.no_grad():
    conv.weight[:] = torch.from_numpy(SOBEL_X)[None, None]
    t = torch.from_numpy(gray)[None, None]                 # (240, 320) → (1, 1, 240, 320)
    edges_torch = conv(t)[0, 0].numpy()
print(f"輸入 tensor {tuple(t.shape)} → 輸出 {tuple(conv(t).shape)}")
print(f"和 cv2.filter2D 比：最大差 {np.abs(edges_torch - edges_cv).max():.6f}")

print("\n== 4. padding：不補邊，每一層都少一圈 ==")
for p in [0, 1]:
    out = nn.Conv2d(1, 1, kernel_size=3, padding=p)(t)
    print(f"padding={p}：{tuple(t.shape[2:])} → {tuple(out.shape[2:])}")
out = nn.Conv2d(1, 1, kernel_size=5, padding=2)(t)
print(f"kernel_size=5、padding=2：{tuple(t.shape[2:])} → {tuple(out.shape[2:])}")

print("\n== 5. in_channels、out_channels：彩色進、好幾張特徵圖出 ==")
rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255
layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
print(f"彩色輸入 {tuple(rgb.shape)} → Conv2d(3, 16, 3) → {tuple(layer(rgb).shape)}")
print(f"weight {tuple(layer.weight.shape)}、bias {tuple(layer.bias.shape)}")

print("\n== 6. 參數量：輸出通道 × 輸入通道 × 核高 × 核寬（＋輸出通道個 bias） ==")
for name, m in [("Conv2d(1, 1, 3, bias=False)", nn.Conv2d(1, 1, 3, bias=False)),
                ("Conv2d(3, 16, 3)", nn.Conv2d(3, 16, 3)),
                ("Conv2d(16, 32, 3)", nn.Conv2d(16, 32, 3)),
                ("Linear(320*240*3, 16)", nn.Linear(320 * 240 * 3, 16))]:
    print(f"{name:<28}{sum(p.numel() for p in m.parameters()):>12,}")

print("\n== 7. 預訓練模型的第一層：核是學出來的 ==")
for name, model in [("resnet18", resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)),
                    ("mobilenet_v3_small", mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1))]:
    first = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    total = sum(p.numel() for p in model.parameters())
    print(f"{name:<20}第一層 {first}")
    print(f"{'':<20}weight {tuple(first.weight.shape)}，{first.weight.numel():,} 個數字；整個模型 {total:,} 個")
w = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).conv1.weight.detach().mean(dim=1)   # 3 個顏色平均
sobel7 = np.zeros((7, 7), np.float32)
sobel7[2:5, 2:5] = SOBEL_X
sim = [float(np.dot(k.numpy().ravel(), sobel7.ravel()) / (np.linalg.norm(k.numpy()) * np.linalg.norm(sobel7))) for k in w]
best = int(np.argmax(np.abs(sim)))
print(f"resnet18 第一層 64 個核，和 Sobel x 最像的是第 {best} 號（相似度 {sim[best]:+.3f}），它中間的 3×3：")
print(w[best, 2:5, 2:5].numpy())
