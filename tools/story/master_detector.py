#@all
"""偵測器：看照片，把「要注意」的畫面變成事件送進平台（ML 課進度存檔 {VERSION}）。

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
#@v17
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
#@v18
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
#@v20
from ultralytics import YOLO
#@v22
import requests
#@v24
import hashlib
#@all

LINE = os.getenv("LINE", "factory")
HERE = Path(__file__).resolve().parent
STORY = HERE.parent
PROFILE = json.loads((STORY / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
CROPS = STORY / "data" / LINE / "crops"
SCENES = STORY / "data" / LINE / "scenes"
NIGHT_CAMERA = {"factory": "camera_04", "care": "camera_14"}[LINE]
EVENT_TYPE = {"factory": "no_helmet", "care": "fall_detected"}[LINE]
#@v17
torch.manual_seed(0)
torch.set_num_threads(4)
#@all


def load_index():
    """小圖清單：檔名、哪一支攝影機、ok（沒事）或 alert（要注意）。"""
    return pd.read_csv(CROPS / "index.csv")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))
#@v17


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
#@v18


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
#@v20


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
#@v21


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
#@v22


def to_events(detections, conf_min):
    """一張圖只要有一個「要注意」的框 ≥ 門檻，就變成一筆 API 課格式的事件。"""
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
    """有設 PLATFORM_URL 就送到那裡（例如 API 課的 server），沒有就啟動替身平台。"""
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
#@v23


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
#@v24


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
#@all


def main():
#@main v16
    index = load_index()
    first = load_rgb(CROPS / index["file"].iloc[0])
    print(f"== {LINE}：{len(index)} 張小圖 ==")
    print(f"一張：shape {first.shape}、dtype {first.dtype}；ok {int((index['label'] == 'ok').sum())} 張、"
          f"alert {int((index['label'] == 'alert').sum())} 張")
    print("\n== 每支攝影機的平均亮度（灰階）==")
    index["brightness"] = [np.asarray(Image.open(CROPS / f).convert("L")).mean() for f in index["file"]]
    print(index.groupby("camera")["brightness"].agg(["count", "mean"]).round(1).to_string())
    print(f"夜間攝影機：{NIGHT_CAMERA}")
    print("\n== 疊成一批：模型要的 (張數, 3, 224, 224) ==")
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    batch = np.stack([np.asarray(Image.open(CROPS / f).convert("RGB").resize((224, 224))) for f in index["file"][:8]])
    batch = ((batch / 255 - mean) / std).astype(np.float32).transpose(0, 3, 1, 2)
    print(f"shape {batch.shape}、dtype {batch.dtype}、範圍 {batch.min():.2f}～{batch.max():.2f}")
#@main v17
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
#@main v18
    index = load_index()
    part = split_by_camera(index)
    y = torch.tensor((index["label"] == "alert").astype(int).to_numpy())
    pixels = np.stack([np.asarray(Image.open(CROPS / f).convert("L").resize((32, 32))) for f in index["file"]])
    x = torch.tensor(pixels.reshape(len(index), -1) / 255, dtype=torch.float32)
    print(f"== {LINE}：直接拿像素訓練（32×32 灰階＝{x.shape[1]} 個數字）==")
    print(part.value_counts().reindex(["train", "valid", "test"]).to_string())
    tr = torch.tensor((part == "train").to_numpy())
    va = torch.tensor((part == "valid").to_numpy())
    model, history = train_linear(x[tr], y[tr], epochs=30, lr=1e-2)
    print("\n== loss 一輪一輪往下掉 ==")
    print("  ".join(f"第{e + 1}輪 {history[e]:.3f}" for e in [0, 4, 9, 19, 29]))
    te = torch.tensor((part == "test").to_numpy())
    for name, mask in [("train", tr), ("valid", va), ("test", te)]:
        acc, rec = accuracy_recall(model, x[mask], y[mask])
        print(f"{name:<6}accuracy {acc:.3f}  recall {rec:.3f}")
    print(f"test 是夜間攝影機 {NIGHT_CAMERA}，這一支攝影機的畫面訓練時完全沒看過")
#@main v19
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
#@main v20
    detector = load_detector()
    print(f"== {LINE}：YOLO 偵測器（類別 {detector.names}）==")
    images = sorted((SCENES / "images" / "test").glob("*.jpg"))[:3]
    print(detect(detector, images, conf=0.25).to_string(index=False))
#@main v21
    detector = load_detector()
    val_pred = detect(detector, sorted((SCENES / "images" / "val").glob("*.jpg")))
    val_truth = load_labels("val")
    print(f"== {LINE}：「要注意」那一類，偵測門檻掃一遍（val）==")
    print(pd.DataFrame([match(val_pred, val_truth, c) for c in [0.9, 0.7, 0.5, 0.3, 0.1]]).to_string(index=False))
    conf, row = choose_conf(val_pred, val_truth, PROFILE["detector_min_recall"])
    print(f"\nrecall 至少 {PROFILE['detector_min_recall']} → 偵測門檻 {conf}")
    test_pred = detect(detector, sorted((SCENES / "images" / "test").glob("*.jpg")))
    print("\n== 用這個門檻打開 test ==")
    print(pd.DataFrame([match(test_pred, load_labels("test"), conf)]).to_string(index=False))
#@main v22
    images = sorted((SCENES / "images" / "test").glob("*.jpg"))
    events = to_events(detect(load_detector(), images), PROFILE["detector_conf"])
    print(f"== {LINE}：test 的 {len(images)} 張照片，{len(events)} 張偵測到「要注意」→ 事件 ==")
    print(json.dumps(events[0], ensure_ascii=False))
    url = platform_url()
    print(f"\n== 送進平台 ==")
    code, body = send(url, events[0], api_key="wrong-key")
    print(f"錯的 API Key → {code} {body}")
    rows = []
    for event in events:
        code, body = send(url, event)
        rows.append({"location": event["location"], "confidence": event["confidence"], "HTTP": code,
                     "alert_status": body["alert_status"] or "（只記錄）"})
    print(pd.DataFrame(rows).head(8).to_string(index=False))
    print("\n" + pd.DataFrame(rows)["alert_status"].value_counts().to_string())
#@main v23 v24
    import judge as judge_module
    if not (judge_module.MODEL_DIR / "judge.json").exists():
        judge_module.train_and_save()
    judge = judge_module.Judge()
    url = platform_url()
    result_rule = run_pipeline(url, judge, use_judge=False)
    url = platform_url()
    result_judge = run_pipeline(url, judge, use_judge=True)
#@main v23
    print(f"== {LINE}：test 的 60 張照片，從攝影機到告警 ==")
    print(pd.DataFrame([{"方法": "平台規則（信心值 0.8／0.5）", **compare(result_rule)},
                        {"方法": "偵測器＋判斷器", **compare(result_judge)}]).to_string(index=False))
    print("\n== 判斷器改變了哪些決定 ==")
    both = result_rule.merge(result_judge, on=["image", "location", "confidence"], suffixes=("_規則", "_判斷器"))
    changed = both[both["alert_status_規則"] != both["alert_status_判斷器"]]
    print(changed.head(10).to_string(index=False))
#@main v24
    manifest, scores = freeze(HERE / "handoff" / LINE, result_rule, result_judge)
    print(f"== {LINE}：交接包 handoff/{LINE}/ ==")
    for name in sorted(p.name for p in (HERE / "handoff" / LINE).iterdir()):
        print(" -", name)
    print("\n" + json.dumps(manifest, ensure_ascii=False, indent=1))
    print("\n" + scores.to_string(index=False))
#@all


if __name__ == "__main__":
    main()
