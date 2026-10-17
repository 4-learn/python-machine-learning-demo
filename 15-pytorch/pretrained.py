"""15｜預訓練 MobileNetV3-Small 對 3 張工地照片推論：ImageNet 的 1000 類裡沒有「安全帽」。"""
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from common import DATA

torch.set_num_threads(4)

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights).eval()   # eval()：推論模式
preprocess = weights.transforms()                    # 縮放、裁切、標準化：和訓練時一樣
categories = weights.meta["categories"]              # 1000 個類別名稱

print("== 1. 這個模型認得哪些「帽子」？ ==")
print("類別總數：", len(categories))
hats = [c for c in categories if "helmet" in c or c.endswith(" hat") or c.endswith(" cap")]
print("名稱含 helmet／hat／cap 的類別：", hats)
print("有沒有 'hard hat'（工地安全帽）？", "hard hat" in categories)


def top5(img):
    x = preprocess(img).unsqueeze(0)          # (3,224,224) → (1,3,224,224)：一批一張
    with torch.no_grad():                     # 推論不需要記錄梯度
        prob = model(x).softmax(dim=1)[0]     # 1000 個分數 → 加總為 1 的機率
    values, ids = prob.topk(5)
    helmet = prob[categories.index("crash helmet")].item()
    rank = int((prob > prob[categories.index("crash helmet")]).sum()) + 1
    return [(categories[i], v.item()) for v, i in zip(values, ids)], helmet, rank


print("\n== 2. 整張照片的前 5 名 ==")
for path in sorted(DATA.glob("photo-*.jpg")):
    img = Image.open(path).convert("RGB")
    pairs, helmet, rank = top5(img)
    print(f"{path.name}（{img.width}×{img.height}）")
    for name, p in pairs:
        print(f"  {name:<18}{p:.3f}")
    print(f"  {'crash helmet':<18}{helmet:.3f}  第 {rank} 名")

print("\n== 3. 只裁出安全帽那一小塊 ==")
CROPS = [  # (照片, 裁切框 x1,y1,x2,y2, 人看到的內容)
    ("photo-site-rebar.jpg", (315, 240, 410, 320), "戴在頭上的黃色安全帽"),
    ("photo-site-rebar.jpg", (650, 290, 745, 370), "戴在頭上的白色安全帽"),
    ("photo-lunch-break.jpg", (20, 420, 170, 530), "放在凳子上的白色安全帽"),
    ("photo-warehouse-night.jpg", (740, 235, 825, 300), "夜間戴在頭上的橘色安全帽"),
]
for name, box, truth in CROPS:
    pairs, helmet, rank = top5(Image.open(DATA / name).convert("RGB").crop(box))
    first, p = pairs[0]
    print(f"{truth:<14}第 1 名 {first}（{p:.3f}）；crash helmet {helmet:.3f}，第 {rank} 名")

print("\n== 4. .pt／.pth 檔裡裝什麼 ==")
state = model.state_dict()
n = sum(v.numel() for v in state.values())
print(f"state_dict：{len(state)} 個 tensor，共 {n:,} 個數字")
for k in list(state)[:3]:
    print(f"  {k:<28}{tuple(state[k].shape)}")
print(f"  ...（最後一層）{list(state)[-1]} {tuple(state[list(state)[-1]].shape)}")
cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
print(f"權重檔 {cached.name}：{cached.stat().st_size / 1e6:.1f} MB")
