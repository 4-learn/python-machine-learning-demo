"""偵測器：看照片，把「要注意」的畫面變成事件送進平台（ML 課進度存檔 v20）。

執行（在這個資料夾裡）：
    LINE=factory python detector.py     # 工廠線：沒戴安全帽
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
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


def fit_probe(x, y):
    """linear probe：預訓練特徵＋ML 課 07 節的邏輯迴歸（先標準化，ML 課 06 節的 Pipeline）。"""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(x, y)


def report(model, x, y):
    pred = model.predict(x)
    return {"accuracy": round(float((pred == y).mean()), 3), "precision": round(precision_score(y, pred, zero_division=0), 3),
            "recall": round(recall_score(y, pred), 3)}


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
