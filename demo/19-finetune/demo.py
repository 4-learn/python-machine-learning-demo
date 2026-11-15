"""19｜torchvision 微調：凍結預訓練骨幹、換掉最後一層、只訓練新的那一層，再存成 state_dict。"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

torch.manual_seed(0)
torch.set_num_threads(4)
CROPS = Path("story/data/factory/crops")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def count(params):
    return sum(p.numel() for p in params)


print("== 1. 模型的最後幾層：classifier ==")
model = mobilenet_v3_small(weights=WEIGHTS).eval()
print(model.classifier)
print(f"全部參數 {count(model.parameters()):,} 個")

print("\n== 2. 凍結骨幹、換掉 classifier[3] ==")
for p in model.parameters():
    p.requires_grad = False
model.classifier[3] = nn.Linear(1024, 2)
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"新的最後一層：{model.classifier[3]}")
print(f"全部參數 {count(model.parameters()):,} 個；要訓練的 {count(trainable):,} 個"
      f"（{count(trainable) / count(model.parameters()):.1%}）")

print("\n== 3. 骨幹只算一次：先把每張圖變成 1024 個數字 ==")
index = pd.read_csv(CROPS / "index.csv")
y = torch.tensor((index["label"] == "alert").astype(int).to_numpy())
preprocess = WEIGHTS.transforms()
with torch.no_grad():
    x = torch.stack([preprocess(Image.open(CROPS / f).convert("RGB")) for f in index["file"]])
    feats = model.classifier[:3](model.avgpool(model.features(x)).flatten(1))
print(f"{len(index)} 張圖 → 特徵 {tuple(feats.shape)}")
with torch.no_grad():
    same = torch.allclose(model(x[:5]), model.classifier[3](feats[:5]), atol=1e-5)
print(f"整個模型跑一次 vs 特徵直接進最後一層，結果相同：{same}")
print(f"訓練 30 輪：每輪重跑骨幹要算 {30 * 200:,} 張次；先算好特徵只要 {len(index)} 張次")


def train_head(head, x, y, epochs=30):
    """18 節的訓練迴圈，只訓練最後一層。"""
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(head(xb), yb).backward()
            optimizer.step()
    return head


def scores(head, x, y):
    with torch.no_grad():
        pred = head(x).argmax(1)
    recall = ((pred == 1) & (y == 1)).sum().item() / (y == 1).sum().item()
    return f"{(pred == y).float().mean().item():.3f}", f"{recall:.3f}"


print("\n== 4. 訓練新的最後一層，骨幹沒有動 ==")
before = model.features[0][0].weight.clone()
head_before = model.classifier[3].weight.clone()
train_mask = torch.tensor(index["camera"].isin(["camera_01", "camera_02"]).to_numpy())
train_head(model.classifier[3], feats[train_mask], y[train_mask])
print(f"骨幹第一層的權重有變嗎：{not torch.equal(before, model.features[0][0].weight)}")
print(f"最後一層的權重有變嗎：{not torch.equal(head_before, model.classifier[3].weight)}")

print("\n== 5. torch.save：只存 state_dict ==")
path = OUT / "head.pt"
torch.save(model.classifier[3].state_dict(), path)
state = torch.load(path)
print(f"head.pt：{path.stat().st_size:,} bytes")
print({k: tuple(v.shape) for k, v in state.items()})
fresh = nn.Linear(1024, 2)
fresh.load_state_dict(state)
with torch.no_grad():
    print(f"讀回來的新物件，答案和原本相同：{torch.equal(fresh(feats), model.classifier[3](feats))}")

print("\n== 6. 怎麼切：隨機切 vs 依攝影機切（夜間 camera_04 當 test）==")
rows = []
cam_train = index["camera"].isin(["camera_01", "camera_02"]).to_numpy()
cam_test = (index["camera"] == "camera_04").to_numpy()
order = torch.randperm(len(index), generator=torch.Generator().manual_seed(0)).numpy()
rnd_train = pd.Series(False, index=index.index)
rnd_train.iloc[order[:200]] = True
rnd_test = pd.Series(False, index=index.index)
rnd_test.iloc[order[300:]] = True
for name, tr, te in [("隨機切（4 支攝影機混在一起）", rnd_train.to_numpy(), rnd_test.to_numpy()),
                     ("依攝影機切（test＝夜間 camera_04）", cam_train, cam_test)]:
    tr, te = torch.tensor(tr), torch.tensor(te)
    torch.manual_seed(0)
    head = train_head(nn.Linear(1024, 2), feats[tr], y[tr])
    acc, rec = scores(head, feats[te], y[te])
    rows.append({"切法": name, "train 張數": int(tr.sum()), "test 張數": int(te.sum()),
                 "test accuracy": acc, "test recall": rec})
print(pd.DataFrame(rows).to_string(index=False))
