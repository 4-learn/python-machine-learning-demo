"""17｜PyTorch：tensor，以及用 ImageNet 預訓練的 MobileNetV3-Small 推論、取特徵。"""
import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

torch.set_num_threads(4)
PHOTO = "story/data/photos/photo-site-rebar.jpg"              # AI 生成的擬真工地照片，不是真實人物

print("== 1. tensor：PyTorch 的陣列 ==")
a = np.zeros((224, 224, 3))
t = torch.from_numpy(a).float().permute(2, 0, 1)
print("NumPy 陣列 ", a.shape, a.dtype)
print("tensor     ", tuple(t.shape), t.dtype, "＝(顏色, 高, 寬)")
print("unsqueeze(0)", tuple(t.unsqueeze(0).shape), "＝一批只有一張")

print("\n== 2. 預訓練模型：載入、推論 ==")
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights).eval()
preprocess = weights.transforms()
categories = weights.meta["categories"]
print(f"參數 {sum(p.numel() for p in model.parameters()):,} 個；答案清單 {len(categories)} 類")
print("名稱裡有 helmet／hat／cap 的：", [c for c in categories if c.endswith(("helmet", " hat", " cap"))])
print("有沒有 'hard hat'？", "hard hat" in categories)
img = Image.open(PHOTO).convert("RGB")
PARTS = [("黃色安全帽", (315, 240, 410, 320)), ("沒戴帽子的頭", (515, 235, 610, 315))]
for name, box in PARTS:
    x = preprocess(img.crop(box)).unsqueeze(0)
    with torch.no_grad():
        prob = model(x).softmax(dim=1)[0]
    top = prob.topk(3)
    names = [categories[i] for i in top.indices.tolist()]
    shown = "、".join(f"{n} {p:.3f}" for n, p in zip(names, top.values.tolist()))
    print(f"{name:<7}{tuple(x.shape)}：{shown}")

print("\n== 3. 拿掉最後一層：每張圖變成 576 個數字 ==")
crops = [preprocess(img.crop(box)) for _, box in PARTS]
with torch.no_grad():
    feats = model.avgpool(model.features(torch.stack(crops))).flatten(1)
print(f"features → avgpool → flatten：{tuple(feats.shape)}，dtype {feats.dtype}")
print("前 5 個數字（黃色安全帽）：", [round(v, 3) for v in feats[0, :5].tolist()])
print("前 5 個數字（沒戴帽子的頭）：", [round(v, 3) for v in feats[1, :5].tolist()])
