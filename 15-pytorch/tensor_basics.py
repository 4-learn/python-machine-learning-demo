"""15｜tensor 與 numpy 對照、nn.Module、裝置（CPU）。"""
import numpy as np
import torch
from torch import nn

from common import FEATURES, load_events

torch.manual_seed(0)

print("== 1. tensor 和 numpy 長得幾乎一樣 ==")
a = np.array([[0.71, 4.3], [0.94, 11.4]])
t = torch.tensor([[0.71, 4.3], [0.94, 11.4]])
print("numpy :", a.shape, a.dtype, "平均", a.mean().round(3))
print("tensor:", tuple(t.shape), t.dtype, "平均", round(t.mean().item(), 3))
print("tensor 第 1 欄（confidence）：", t[:, 0].tolist())

print("\n== 2. 互相轉換：from_numpy 共用同一塊記憶體 ==")
b = torch.from_numpy(a)
a[0, 0] = 0.99
print("改了 numpy 之後，tensor 也變了：", b[0, 0].item())
print("dtype：numpy 預設", b.dtype, "；模型要的是", b.float().dtype)

print("\n== 3. 一張照片、一批照片、一列事件都是 tensor ==")
photo = torch.zeros(3, 224, 224)
batch = torch.zeros(8, 3, 224, 224)
events = load_events()
row = torch.tensor(events[["confidence", "duration_sec", "light_lux"]].head(4).to_numpy(), dtype=torch.float32)
print("一張彩色照片    ", tuple(photo.shape), "＝(顏色, 高, 寬)")
print("8 張照片一批    ", tuple(batch.shape), "＝(張數, 顏色, 高, 寬)")
print("4 筆事件 3 個欄位", tuple(row.shape), "＝(筆數, 欄位)")


print("\n== 4. nn.Module：模型就是「有參數的函式」==")


class NotifyNet(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.hidden = nn.Linear(n_in, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


model = NotifyNet(n_in=3)
print(model)
n = sum(p.numel() for p in model.parameters())
print(f"參數 {n} 個 = 3×8+8（hidden）+ 8×1+1（out）")
with torch.no_grad():
    print("還沒訓練的輸出：", [round(v, 3) for v in model(row).squeeze(1).tolist()])

print("\n== 5. 裝置：這台電腦用 CPU ==")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch 版本：", torch.__version__)
print("有 GPU（CUDA）嗎？", torch.cuda.is_available(), "→ device =", device)
print("模型參數放在：", next(model.parameters()).device)
