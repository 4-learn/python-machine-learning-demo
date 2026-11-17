"""17｜PyTorch：tensor、nn.Module，以及用 ImageNet 預訓練的 MobileNetV3-Small 推論。"""
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

torch.manual_seed(0)
torch.set_num_threads(4)
PHOTO = "story/data/photos/photo-site-rebar.jpg"              # AI 生成的擬真工地照片，不是真實人物

print("== 1. tensor 和 NumPy 陣列長得幾乎一樣 ==")
a = np.array([[0.71, 4.3], [0.94, 11.4]])
t = torch.tensor([[0.71, 4.3], [0.94, 11.4]])
print("NumPy :", a.shape, a.dtype, "平均", round(float(a.mean()), 3))
print("tensor:", tuple(t.shape), t.dtype, "平均", round(t.mean().item(), 3))
print("tensor 第 0 欄：", t[:, 0].tolist(), "← float32 存不下那麼多位數")

print("\n== 2. from_numpy：共用同一塊記憶體 ==")
b = torch.from_numpy(a)
a[0, 0] = 0.99
print("改了 NumPy 之後，tensor 也變了：", b[0, 0].item())
print("dtype：從 NumPy 來的是", b.dtype, "；.float() 之後是", b.float().dtype)

print("\n== 3. shape 就是資料的意思 ==")
print("一張彩色照片     ", tuple(torch.zeros(3, 224, 224).shape), "＝(顏色, 高, 寬)")
print("8 張照片一批     ", tuple(torch.zeros(8, 3, 224, 224).shape), "＝(張數, 顏色, 高, 寬)")
print("unsqueeze(0) 之後", tuple(torch.zeros(3, 224, 224).unsqueeze(0).shape), "＝一批只有一張")

print("\n== 4. nn.Module：模型就是「有參數的函式」==")


class TinyNet(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.hidden = nn.Linear(n_in, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))


tiny = TinyNet(n_in=3)
print(tiny)
print(f"參數 {sum(p.numel() for p in tiny.parameters())} 個 = 3×4+4（hidden）+ 4×2+2（out）")
with torch.no_grad():
    print("還沒訓練就給答案：", [round(v, 3) for v in tiny(torch.tensor([[0.9, 30.0, 500.0]]))[0].tolist()])

print("\n== 5. 預訓練模型：MobileNetV3-Small（ImageNet）==")
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights).eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]
print(f"參數 {sum(p.numel() for p in model.parameters()):,} 個；答案清單 {len(categories)} 類")
print("名稱裡有 helmet／hat／cap 的：", [c for c in categories if c.endswith(("helmet", " hat", " cap"))])
for word in ["hard hat", "fall", "fallen"]:
    print(f"有沒有 {word!r}？", word in categories)

print("\n== 6. 推論：eval、no_grad、softmax、topk ==")
img = Image.open(PHOTO).convert("RGB")
# box 是 None 時 img.crop(None) 就是整張照片
PARTS = [("整張照片", None), ("黃色安全帽", (315, 240, 410, 320)), ("沒戴帽子的頭", (515, 235, 610, 315))]
for name, box in PARTS:
    x = preprocess(img.crop(box)).unsqueeze(0)
    with torch.no_grad():
        prob = model(x).softmax(dim=1)[0]
    top = prob.topk(3)
    names = [categories[i] for i in top.indices.tolist()]
    shown = "、".join(f"{n} {p:.3f}" for n, p in zip(names, top.values.tolist()))
    print(f"{name:<7}{tuple(x.shape)}：{shown}")

print("\n== 7. 拿掉最後一層：每張圖變成 576 個數字 ==")
print("最後幾層：", model.classifier)
crops = [preprocess(img.crop(box)) for _, box in PARTS[1:]]
with torch.no_grad():
    feats = model.avgpool(model.features(torch.stack(crops))).flatten(1)
print(f"features → avgpool → flatten：{tuple(feats.shape)}，dtype {feats.dtype}")
print("前 5 個數字（黃色安全帽）：", [round(v, 3) for v in feats[0, :5].tolist()])

print("\n== 8. 裝置：這台電腦用 CPU ==")
print("torch 版本：", torch.__version__)
print("有 GPU（CUDA）嗎？", torch.cuda.is_available())
print("模型參數放在：", next(model.parameters()).device)
