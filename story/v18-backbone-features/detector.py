"""偵測器：看照片，把「要注意」的畫面變成事件送進平台（ML 課進度存檔 v18）。

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
from sklearn.neighbors import KNeighborsClassifier
from torchvision.models import get_model, get_model_weights

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


BACKBONES = ["mobilenet_v3_small", "resnet18"]


def backbone_features(name, paths, batch=32):
    """用 torchvision 的名字載入 ImageNet 預訓練模型，拿掉最後一層，每張圖變成一串數字。"""
    weights = get_model_weights(name)["IMAGENET1K_V1"]
    model = get_model(name, weights=weights).eval()
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()          # ResNet 的最後一層叫 fc
    else:
        model.classifier = torch.nn.Identity()  # MobileNet 的最後幾層叫 classifier
    pre = weights.transforms()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            out.append(model(torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])))
    return torch.cat(out).numpy()


def pixel_features(paths, size=32):
    """對照組：32×32 灰階像素直接攤平。"""
    return np.stack([np.asarray(Image.open(p).convert("L").resize((size, size)), np.float32).ravel() / 255 for p in paths])


def knn_accuracy(x, y, part, split):
    """ML 課 05 節的 KNN（k=1）：最像的那張小圖是什麼，就猜什麼。"""
    knn = KNeighborsClassifier(n_neighbors=1).fit(x[part == "train"], y[part == "train"])
    return float((knn.predict(x[part == split]) == y[part == split]).mean())


def choose_backbone(index, part, y):
    """在 valid 上比較每一個骨幹（KNN 準確率），選最高的；test 不看。"""
    paths = [CROPS / f for f in index["file"]]
    scores = {name: knn_accuracy(backbone_features(name, paths), y, part, "valid") for name in BACKBONES}
    return max(scores, key=scores.get), scores


def main():
    index = load_index()
    part = split_by_camera(index).to_numpy()
    y = (index["label"] == "alert").astype(int).to_numpy()
    paths = [CROPS / f for f in index["file"]]
    print(f"== {LINE}：三種「把小圖變成數字」的方法，都用 KNN（k=1）打成績 ==")
    rows = []
    for name, x in [("像素 32×32", pixel_features(paths))] + [(n, backbone_features(n, paths)) for n in BACKBONES]:
        rows.append({"特徵": name, "數字個數": x.shape[1], "valid": knn_accuracy(x, y, part, "valid"),
                     "test（夜間）": knn_accuracy(x, y, part, "test")})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    best, scores = choose_backbone(index, part, y)
    print(f"\n只看 valid 選骨幹：{best}（valid {scores[best]:.3f}）")


if __name__ == "__main__":
    main()
