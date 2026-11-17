"""18｜torchvision get_model：用名字載入不同的預訓練骨幹，拿掉最後一層變成特徵，再用 05 節的 KNN 比較。"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchvision.models import get_model, get_model_weights, list_models

torch.manual_seed(0)
torch.set_num_threads(4)
CROPS = Path("story/data/factory/crops")
BACKBONES = ["mobilenet_v3_small", "resnet18"]

rows = []
names = list_models(module=torchvision.models)
for name in ["mobilenet_v3_small", "resnet18", "efficientnet_b0", "resnet50"]:
    enum = get_model_weights(name)
    meta = enum["IMAGENET1K_V1"].meta
    rows.append({"名字": name, "可用的權重": "、".join(w.name for w in enum),
                 "參數（百萬）": round(meta["num_params"] / 1e6, 1),
                 "ImageNet top-1（%）": meta["_metrics"]["ImageNet-1K"]["acc@1"]})
print("== 1. list_models：torchvision 內建幾種分類模型 ==")
print(f"共 {len(names)} 種；名字以 resnet 開頭的：{[n for n in names if n.startswith('resnet')]}")
print("\n== 2. get_model_weights：先看規格，不必下載 ==")
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 3. get_model：用名字載入，每個模型的最後一層名字不一樣 ==")
models, preprocess = {}, {}
for name in ["mobilenet_v3_small", "resnet18"]:
    weights = get_model_weights(name)["IMAGENET1K_V1"]
    models[name] = get_model(name, weights=weights).eval()
    preprocess[name] = weights.transforms()
    last_name, last = list(models[name].named_children())[-1]
    print(f"{name:<19} 最後一層叫 {last_name}：{type(last).__name__}，輸出 {len(weights.meta['categories'])} 類")

print("\n== 4. nn.Identity：把最後一層換成「原樣交出去」==")
index = pd.read_csv(CROPS / "index.csv")
one = Image.open(CROPS / index["file"].iloc[0]).convert("RGB")
for name, model in models.items():
    x = preprocess[name](one).unsqueeze(0)
    with torch.no_grad():
        before = tuple(model(x).shape)
        if name.startswith("resnet"):
            model.fc = nn.Identity()
        else:
            model.classifier = nn.Identity()
        after = tuple(model(x).shape)
    print(f"{name:<19} 換之前 {before} → 換之後 {after}")
wrong = get_model("resnet18", weights="IMAGENET1K_V1").eval()
wrong.classifier = nn.Identity()                 # resnet18 沒有 classifier
with torch.no_grad():
    print(f"resnet18 寫成 classifier = Identity：輸出還是 {tuple(wrong(x).shape)}，不會報錯")


def pixel_features(paths, size=32):
    """對照組：32×32 灰階像素直接攤平。"""
    return np.stack([np.asarray(Image.open(p).convert("L").resize((size, size)), np.float32).ravel() / 255
                     for p in paths])


def backbone_features(model, pre, paths, batch=50):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            out.append(model(torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])))
    return torch.cat(out).numpy()


def knn_accuracy(x):
    knn = KNeighborsClassifier(n_neighbors=1).fit(x[learn], y[learn])
    accuracy = (knn.predict(x[~learn]) == y[~learn]).mean()
    return round(float(accuracy), 3)


print("\n== 5. 同一把尺：KNN（k=1）在 camera_01 學、考 camera_02 ==")
small = index[index["camera"].isin(["camera_01", "camera_02"])].reset_index(drop=True)
paths = [CROPS / f for f in small["file"]]
y = (small["label"] == "alert").to_numpy()
learn = (small["camera"] == "camera_01").to_numpy()
feats = {"像素 32×32": pixel_features(paths)}
feats.update({name: backbone_features(models[name], preprocess[name], paths) for name in BACKBONES})
rows = [{"特徵": k, "數字個數": x.shape[1], "accuracy": knn_accuracy(x)} for k, x in feats.items()]
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))
