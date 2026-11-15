"""偵測器：看照片，把「要注意」的畫面變成事件送進平台（ML 課進度存檔 v17）。

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


def main():
    index = load_index()
    model = load_backbone()
    categories = WEIGHTS.meta["categories"]
    print(f"== {LINE}：ImageNet 預訓練模型認得這些小圖嗎？ ==")
    for label in ["ok", "alert"]:
        path = CROPS / index[index["label"] == label]["file"].iloc[0]
        with torch.no_grad():
            prob = model(PREPROCESS(Image.open(path).convert("RGB")).unsqueeze(0)).softmax(1)[0]
        top = prob.topk(3)
        names = "、".join(f"{categories[i]} {p:.2f}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"{label:<6}{path.name}：{names}")
    print("\n== 拿掉最後一層：每張圖變成一串數字 ==")
    x = features(model, [CROPS / f for f in index["file"]])
    print(f"特徵 shape {tuple(x.shape)}、dtype {x.dtype}")
    params = sum(p.numel() for p in model.parameters())
    print(f"這個模型有 {params:,} 個參數；最後一層（1000 類）有 {sum(p.numel() for p in model.classifier[3].parameters()):,} 個")


if __name__ == "__main__":
    main()
