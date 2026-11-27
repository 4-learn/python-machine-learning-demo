"""偵測器：看照片，把「要注意」的畫面變成事件送進平台（ML 課進度存檔 v24）。

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
import requests
import hashlib

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


def iou(a, b):
    """兩個框 (x1, y1, x2, y2) 重疊的面積 ÷ 合起來的面積。"""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def load_labels(split):
    """YOLO 標註（中心、寬高、0–1）換回像素的 (x1, y1, x2, y2)。"""
    rows = []
    for path in sorted((SCENES / "labels" / split).glob("*.txt")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            cls, cx, cy, w, h = map(float, line.split())
            rows.append({"image": path.stem + ".jpg", "cls": int(cls),
                         "x1": (cx - w / 2) * 320, "y1": (cy - h / 2) * 320, "x2": (cx + w / 2) * 320, "y2": (cy + h / 2) * 320})
    return pd.DataFrame(rows)


def match(pred, truth, conf, iou_min=0.5):
    """只看「要注意」那一類：信心值 ≥ conf 的框，和標註框 IoU ≥ 0.5 才算抓到。"""
    p = pred[(pred["cls"] == 1) & (pred["conf"] >= conf)].sort_values("conf", ascending=False)
    t = truth[truth["cls"] == 1]
    used, tp = set(), 0
    for _, box in p.iterrows():
        cands = t[t["image"] == box["image"]]
        best, best_i = 0.0, None
        for i, gt in cands.iterrows():
            if i in used:
                continue
            v = iou(box[["x1", "y1", "x2", "y2"]].tolist(), gt[["x1", "y1", "x2", "y2"]].tolist())
            if v > best:
                best, best_i = v, i
        if best >= iou_min:
            used.add(best_i)
            tp += 1
    fp, fn = len(p) - tp, len(t) - tp
    return {"conf": conf, "TP": tp, "FP": fp, "FN": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0, "recall": round(tp / (tp + fn), 3)}


def choose_conf(pred, truth, min_recall):
    """在 val 上，由高往低找：第一個讓 recall 達標的偵測門檻。"""
    for conf in np.round(np.arange(0.9, 0.04, -0.05), 2):
        row = match(pred, truth, conf)
        if row["recall"] >= min_recall:
            return float(conf), row
    return 0.05, match(pred, truth, 0.05)


def to_events(detections, conf_min):
    """一張圖只要有一個「要注意」的框 ≥ 門檻，就變成一筆平台格式的事件。"""
    meta = pd.read_csv(SCENES / "meta.csv").set_index("image")
    hits = detections[(detections["cls"] == 1) & (detections["conf"] >= conf_min)]
    events = []
    for image, group in hits.groupby("image"):
        m = meta.loc[image]
        events.append({"event_type": EVENT_TYPE, "location": m["location"], "source": m["source"],
                       "confidence": float(group["conf"].max()), "occurred_at": m["occurred_at"].replace(" ", "T"),
                       "detail": {"duration_seconds": float(m["duration_seconds"]), "light_lux": int(m["light_lux"]),
                                  "person_id": m["person_id"], "image": image}})
    return events


def platform_url():
    """有設 PLATFORM_URL 就送到那裡（例如你自己架的平台），沒有就啟動替身平台。"""
    url = os.getenv("PLATFORM_URL")
    if url:
        return url
    import sys
    sys.path.insert(0, str(STORY))
    import platform_mock
    return platform_mock.start(PROFILE)


def send(url, event, api_key="dev-device"):
    r = requests.post(f"{url}/events", json=event, headers={"X-API-Key": api_key}, timeout=5)
    return r.status_code, r.json()


def run_pipeline(url, judge, use_judge):
    """test 的 60 張照片：偵測 → 事件 → （判斷器）→ 送進平台。"""
    conf_min = PROFILE["detector_conf"]
    images = sorted((SCENES / "images" / "test").glob("*.jpg"))
    events = to_events(detect(load_detector(), images), conf_min)
    rows = []
    for event in events:
        if use_judge:
            event["detail"]["judge"] = judge.decide(event)
        code, body = send(url, event)
        rows.append({"image": event["detail"]["image"], "location": event["location"],
                     "confidence": event["confidence"], "alert_status": body["alert_status"] or "（只記錄）"})
    return pd.DataFrame(rows)


def compare(result):
    """和事後確認的結果比：每張照片該不該叫人（meta.csv 的 should_alert）。"""
    meta = pd.read_csv(SCENES / "meta.csv")
    called = set(result.loc[result["alert_status"] == "open", "image"])
    review = set(result.loc[result["alert_status"] == "pending_review", "image"])
    should = set(meta.loc[meta["should_alert"] == 1, "image"])
    return {"馬上通知": len(called), "其中該叫": len(called & should), "待確認": len(review),
            "漏掉（該叫卻只記錄或沒偵測到）": len(should - called - review)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(out_dir, result_rule, result_judge):
    """交接包：考卷（test 照片的清單與答案）、指紋、成績表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(SCENES / "meta.csv")
    exam = meta[["image", "source", "location", "occurred_at", "duration_seconds", "light_lux", "should_alert"]]
    exam.to_csv(out_dir / "eval-set.csv", index=False)
    scores = pd.DataFrame([{"方法": "平台規則（信心值 0.8／0.5）", **compare(result_rule)},
                           {"方法": "偵測器＋判斷器", **compare(result_judge)}])
    scores.to_csv(out_dir / "scores.csv", index=False)
    manifest = {"line": LINE, "images": len(exam), "should_alert": int(exam["should_alert"].sum()),
                "eval_set_sha256": sha256(out_dir / "eval-set.csv"),
                "detector_sha256_12": sha256(STORY.parent / "models" / f"{LINE}-yolo11n.pt")[:12],
                "detector_conf": PROFILE["detector_conf"]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest, scores


def main():
    import judge as judge_module
    if not (judge_module.MODEL_DIR / "judge.json").exists():
        judge_module.train_and_save()
    judge = judge_module.Judge()
    url = platform_url()
    result_rule = run_pipeline(url, judge, use_judge=False)
    url = platform_url()
    result_judge = run_pipeline(url, judge, use_judge=True)
    manifest, scores = freeze(HERE / "handoff" / LINE, result_rule, result_judge)
    print(f"== {LINE}：交接包 handoff/{LINE}/ ==")
    for name in sorted(p.name for p in (HERE / "handoff" / LINE).iterdir()):
        print(" -", name)
    print("\n" + json.dumps(manifest, ensure_ascii=False, indent=1))
    print("\n" + scores.to_string(index=False))


if __name__ == "__main__":
    main()
