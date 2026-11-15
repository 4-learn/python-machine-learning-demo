"""偵測器：看照片，把「要注意」的畫面變成事件送進平台（ML 課進度存檔 v19）。

執行（在這個資料夾裡）：
    LINE=factory python detector.py     # 工廠線：沒戴安全帽
    LINE=care python detector.py        # 健康線：倒在地上
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LINE = os.getenv("LINE", "factory")
HERE = Path(__file__).resolve().parent
STORY = HERE.parent
PROFILE = json.loads((STORY / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
CROPS = STORY / "data" / LINE / "crops"
SCENES = STORY / "data" / LINE / "scenes"
NIGHT_CAMERA = {"factory": "camera_04", "care": "camera_14"}[LINE]
EVENT_TYPE = {"factory": "no_helmet", "care": "fall_detected"}[LINE]
torch.manual_seed(0)
torch.set_num_threads(4)


def load_index():
    """小圖清單：檔名、哪一支攝影機、ok（沒事）或 alert（要注意）。"""
    return pd.read_csv(CROPS / "index.csv")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1
PREPROCESS = WEIGHTS.transforms()


def load_backbone():
    """ImageNet 預訓練的 MobileNetV3-Small，只拿來看圖、不訓練。"""
    return mobilenet_v3_small(weights=WEIGHTS).eval()


def features(model, paths):
    """每張圖變成 576 個數字（拿掉最後的分類層）。"""
    with torch.no_grad():
        x = torch.stack([PREPROCESS(Image.open(p).convert("RGB")) for p in paths])
        return model.avgpool(model.features(x)).flatten(1)


def split_by_camera(index):
    """依攝影機切：夜間那支當 test，另一支當 valid，其餘兩支拿來學。"""
    cams = [c for c in sorted(index["camera"].unique()) if c != NIGHT_CAMERA]
    valid_cam = cams[-1]
    part = np.where(index["camera"] == NIGHT_CAMERA, "test", np.where(index["camera"] == valid_cam, "valid", "train"))
    return pd.Series(part, index=index.index)


def train_linear(x, y, epochs, lr=1e-3, batch_size=32):
    """最簡單的訓練迴圈：一層 nn.Linear，CrossEntropyLoss＋Adam。"""
    model = nn.Linear(x.shape[1], 2)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        history.append(total / len(x))
    return model, history


def accuracy_recall(model, x, y):
    with torch.no_grad():
        pred = model(x).argmax(1)
    tp = int(((pred == 1) & (y == 1)).sum())
    return float((pred == y).float().mean()), tp / int((y == 1).sum())


def main():
    index = load_index()
    part = split_by_camera(index)
    y = torch.tensor((index["label"] == "alert").astype(int).to_numpy())
    x = features(load_backbone(), [CROPS / f for f in index["file"]])
    masks = {p: torch.tensor((part == p).to_numpy()) for p in ["train", "valid", "test"]}
    head, history = train_linear(x[masks["train"]], y[masks["train"]], epochs=30)
    print(f"== {LINE}：預訓練特徵＋重新訓練最後一層 ==")
    print("  ".join(f"第{e + 1}輪 {history[e]:.3f}" for e in [0, 4, 9, 19, 29]))
    for name in ["train", "valid", "test"]:
        acc, rec = accuracy_recall(head, x[masks[name]], y[masks[name]])
        print(f"{name:<6}accuracy {acc:.3f}  recall {rec:.3f}")
    cams = index.loc[part == "test", "camera"].iloc[0]
    print(f"test 是夜間攝影機 {cams}：模型沒看過這麼暗的畫面")
    out = HERE / "out" / LINE
    out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out / "head.pt")
    print(f"已存 out/{LINE}/head.pt（只存最後一層，{sum(p.numel() for p in head.parameters()):,} 個參數）")


if __name__ == "__main__":
    main()
