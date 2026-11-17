"""28｜torchvision resnet18.conv1：預訓練第一層學到什麼。看權重 shape → 手寫 Sobel x → 比相關係數 → 隨機權重對照 → stride、padding → 特徵圖 → 64 個濾波器拼成一張圖。"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
np.set_printoptions(precision=3, suppress=True)

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

print("\n== 3. 64 個學出來的濾波器，哪幾個像 Sobel x ==")
gray_k = w.mean(dim=1).numpy()                             # 三個顏色平均 → (64, 7, 7)
small = np.stack([cv2.resize(k, (3, 3), interpolation=cv2.INTER_AREA) for k in gray_k])
r = np.array([np.corrcoef(k.ravel(), SOBEL_X.ravel())[0, 1] for k in small])
size = w.abs().amax(dim=(1, 2, 3)).numpy()                 # 每個濾波器最大的 |權重|
for i in np.argsort(-np.abs(r))[:3]:
    print(f"第 {i:>2} 個：相關係數 {r[i]:+.3f}；最大 |權重| {size[i]:.3f}")
dead = size < 1e-3
print(f"最大 |權重| < 0.001 的「沒在用」濾波器 {dead.sum()} 個：{np.where(dead)[0].tolist()}；其他的中位數 {np.median(size[~dead]):.3f}")
alive_r = np.where(dead, 0, r)
print(f"只看有在用的 {(~dead).sum()} 個：|相關係數| ≥ 0.8 的 {(np.abs(alive_r) >= 0.8).sum()} 個、≥ 0.5 的 {(np.abs(alive_r) >= 0.5).sum()} 個")
r_y = np.where(dead, 0, [np.corrcoef(k.ravel(), SOBEL_X.T.ravel())[0, 1] for k in small])
print(f"換成 Sobel y（找水平的邊）：最像的是第 {np.argmax(np.abs(r_y))} 個，{r_y[np.argmax(np.abs(r_y))]:+.3f}")
best = int(np.argmax(np.abs(r)))
print(f"第 {best} 個（三色平均）7×7：\n", gray_k[best])
print("縮成 3×3：\n", small[best])

print("\n== 4. 對照組：沒訓練過的 resnet18（隨機權重）==")
torch.manual_seed(0)
w_rand = resnet18(weights=None).conv1.weight.detach()     # 同樣的形狀，數字是亂數
small_rand = [cv2.resize(k, (3, 3), interpolation=cv2.INTER_AREA) for k in w_rand.mean(dim=1).numpy()]
r_rand = np.array([np.corrcoef(k.ravel(), SOBEL_X.ravel())[0, 1] for k in small_rand])
print(f"縮成 3×3 比：隨機權重最像的也有 {np.abs(r_rand).max():.3f}（只有 9 個數字，碰巧像不難）")
sobel7 = cv2.resize(SOBEL_X, (7, 7), interpolation=cv2.INTER_LINEAR)   # 把 Sobel 放大成 7×7，49 個數字一起比
r7 = np.array([np.corrcoef(k.ravel(), sobel7.ravel())[0, 1] if not d else 0 for k, d in zip(gray_k, dead)])
r7_rand = np.array([np.corrcoef(k.ravel(), sobel7.ravel())[0, 1] for k in w_rand.mean(dim=1).numpy()])
print(f"放大成 7×7 比：預訓練第 {np.argmax(np.abs(r7))} 個 {r7[np.argmax(np.abs(r7))]:+.3f}；隨機權重最像的只有 {np.abs(r7_rand).max():.3f}")

print("\n== 5. stride、padding：輸出多大 ==")
img = cv2.imread(str(ROOT / "story" / "data" / "factory" / "roi" / "frames" / "frame_000.jpg"))
rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
prep = weights.transforms()                                # 17 節的前處理；這裡只借它的平均、標準差
mean = torch.tensor(prep.mean)[:, None, None]
std = torch.tensor(prep.std)[:, None, None]
x = ((rgb - mean) / std)[None]                             # (1, 3, 240, 320)，不裁切，座標對得回原圖
with torch.no_grad():
    maps = conv(x)
    print(f"conv1（stride 2、padding 3）：{tuple(x.shape)} → {tuple(maps.shape)}")
    for s, p in [(1, 3), (2, 0)]:
        print(f"同樣的權重，stride {s}、padding {p}：→ {tuple(F.conv2d(x, w, stride=s, padding=p).shape)}")
print(f"公式：(240 + 2×3 − 7) // 2 + 1 = {(240 + 2 * 3 - 7) // 2 + 1}；(320 + 2×3 − 7) // 2 + 1 = {(320 + 2 * 3 - 7) // 2 + 1}")

print(f"\n== 6. 特徵圖：第 {best} 個在畫面上看到什麼 ==")
for line in ["factory", "care"]:
    frame = cv2.imread(str(ROOT / "story" / "data" / line / "roi" / "frames" / "frame_000.jpg"))
    t = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        fmaps = conv(((t - mean) / std)[None])[0]          # (64, 120, 160)
    col = fmaps[best].numpy().mean(axis=0)[3:-3]           # 每一行（x）平均；去掉邊上補 0 造成的假邊
    hi, lo = int(np.argmax(col)) + 3, int(np.argmin(col)) + 3
    print(f"{line:<8}特徵圖 {tuple(fmaps[best].shape)}；最正的一行 x={hi}（原圖約 x={hi * 2}）{col[hi - 3]:+.3f}、"
          f"最負的一行 x={lo}（原圖約 x={lo * 2}）{col[lo - 3]:+.3f}")
strength = maps[0].abs().mean(dim=(1, 2)).numpy()
top3, low3 = np.argsort(-strength)[:3], np.argsort(strength)[:3]
print(f"工廠畫面 64 張特徵圖，平均反應最強的 3 個：{top3.tolist()}；最弱的 3 個：{low3.tolist()}")
print("權重加總（不是 0 就會對整片顏色有反應，Sobel x 是 0）：" + "、".join(f"第 {i} 個 {w[i].sum():+.3f}" for i in [*top3, best]))

print("\n== 7. 把 64 個濾波器拼成一張圖 ==")
OUT.mkdir(exist_ok=True)
tiles = []
for k, m in zip(w.numpy(), size):                          # (3, 7, 7)；0 畫成灰色，正的亮、負的暗
    k = (k / max(m, 1e-3) * 0.5 + 0.5) * 255                # 各自放大到看得見；沒在用的（幾乎全 0）維持灰色
    tile = cv2.resize(k.transpose(1, 2, 0).astype(np.uint8), (35, 35), interpolation=cv2.INTER_NEAREST)
    tiles.append(cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
grid = cv2.cvtColor(np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 64, 8)]), cv2.COLOR_RGB2BGR)
cv2.imwrite(str(OUT / "conv1-filters.png"), grid)
print(f"存到 demo/28-first-layer/out/conv1-filters.png：{grid.shape}（8×8 格，每格一個濾波器放大 5 倍）")
