"""偵測器：看照片，把「要注意」的畫面變成事件送進平台（ML 課進度存檔 v20）。

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
from ultralytics import YOLO

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


def load_detector():
    """教師用合成場景訓練好的 YOLO11n（20 節）：類別 0 沒事、1 要注意。"""
    return YOLO(str(STORY.parent / "models" / f"{LINE}-yolo11n.pt"))


def detect(detector, image_paths, conf=0.01):
    """每個偵測框一列：哪張圖、類別、信心值、框的四個角。"""
    rows = []
    for result in detector.predict([str(p) for p in image_paths], conf=conf, imgsz=320, device="cpu",
                                   verbose=False, stream=True):
        name = Path(result.path).name
        for cls, score, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
            rows.append({"image": name, "cls": int(cls), "conf": round(score, 3),
                         "x1": round(box[0], 1), "y1": round(box[1], 1), "x2": round(box[2], 1), "y2": round(box[3], 1)})
    return pd.DataFrame(rows, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"])


def main():
    detector = load_detector()
    print(f"== {LINE}：YOLO 偵測器（類別 {detector.names}）==")
    images = sorted((SCENES / "images" / "test").glob("*.jpg"))[:3]
    print(detect(detector, images, conf=0.25).to_string(index=False))


if __name__ == "__main__":
    main()
