#@all
"""進階：攝影機 ROI、骨架、串流資料、判斷器進階（ML 課進度存檔 {VERSION}，AI 新秀）。

執行（在這個資料夾裡）：
    LINE=factory python extra.py     # 工廠線：通道被占用、彎腰搬重物
    LINE=care python extra.py        # 健康線：夜間離床、跌倒
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
#@v25
import cv2
#@v27
import requests
#@v28
import torch
from torch import nn
#@v29
from ultralytics import YOLO
#@v30
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
#@v35
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#@v36
from sklearn.model_selection import TimeSeriesSplit
#@v37
from sklearn.model_selection import StratifiedGroupKFold, learning_curve
#@v38
from scipy.stats import loguniform, randint
from sklearn.model_selection import RandomizedSearchCV
#@v39
from sklearn.utils.class_weight import compute_class_weight
#@v40
from sklearn.decomposition import PCA
#@all

LINE = os.getenv("LINE", "factory")
HERE = Path(__file__).resolve().parent
STORY = HERE.parent
PROFILE = json.loads((STORY / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
DATA = STORY / "data" / LINE
#@v25


def load_roi():
    """攝影機畫面清單與 ROI（設定在 roi.json：哪一塊區域要看）。"""
    index = pd.read_csv(DATA / "roi" / "index.csv")
    roi = json.loads((DATA / "roi" / "roi.json").read_text(encoding="utf-8"))
    return index, roi


def read_frame(name):
    img = cv2.imread(str(DATA / "roi" / "frames" / name))
    if img is None:
        raise FileNotFoundError(name)
    return img


def crop_roi(img, roi):
    return img[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
#@v26


def roi_gray(img, roi):
    """ROI 轉灰階、模糊去雜訊，再把亮度拉到同一個水準（夜間畫面整體比較暗）。"""
    gray = cv2.cvtColor(cv2.GaussianBlur(crop_roi(img, roi), (5, 5), 0), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.convertScaleAbs(gray * (128 / max(float(np.median(gray)), 1.0)))


def roi_score(img, roi, reference, diff_min=15):
    """ROI 裡「多出來的東西」有多大：和空 ROI 相減 → 二值化 → 找輪廓 → 最大一塊占 ROI 幾成。"""
    diff = cv2.absdiff(roi_gray(img, roi), reference)
    _, mask = cv2.threshold(diff, diff_min, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max((cv2.contourArea(c) for c in contours), default=0.0)
    return biggest / mask.size


def is_empty(row, roi):
    """這張畫面的 ROI 是不是空的（工廠：通道沒貨物；健康：床上沒人）。"""
    return row["label"] == (1 if roi["alert_when"] == "empty" else 0)


def make_reference(index, roi):
    """參考畫面：白天、ROI 空著的畫面取中位數。"""
    empty = index[(index["night"] == 0) & index.apply(is_empty, axis=1, roi=roi)]["file"][:15]
    return np.median(np.stack([roi_gray(read_frame(f), roi) for f in empty]), axis=0).astype(np.uint8)


def decide(score, roi):
    """ROI 被占用（score ≥ 門檻）時，依設定決定要不要發事件。"""
    occupied = score >= PROFILE["roi_score_min"]
    return occupied if roi["alert_when"] == "occupied" else not occupied
#@v27


class Camera:
    """攝影機：有設 EZCAM_URL 就向它要一張照片；沒有就從本機畫面依序讀（Mock 模式）。"""

    def __init__(self, index):
        self.url = os.getenv("EZCAM_URL")
        self.files = list(index["file"])
        self.i = 0

    def capture(self):
        if self.url:
            r = requests.get(f"{self.url}/capture", timeout=10)
            r.raise_for_status()
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            return "ezcam", img
        name = self.files[self.i % len(self.files)]
        self.i += 1
        data = (DATA / "roi" / "frames" / name).read_bytes()
        return name, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
#@v28


def edge_kernel():
    """一個 3×3 的「找垂直邊」濾波器，放進 nn.Conv2d。"""
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])
    return conv
#@v29


KPTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]


def load_pose_model():
    return YOLO(str(STORY.parent / "models" / "yolo11n-pose.pt"))


def keypoints(model, image_path):
    """一張照片 → (人數, 17, 2) 的座標與 (人數, 17) 的信心值。"""
    result = model.predict(str(image_path), imgsz=640, device="cpu", verbose=False)[0]
    return result.keypoints.xy.numpy(), result.keypoints.conf.numpy()
#@v30


def load_poses(name="poses.csv"):
    return pd.read_csv(DATA / "pose" / name, keep_default_na=False)


def frame_features(df):
    """每一格骨架 → 3 個和畫面大小、站的位置無關的數字。"""
    sx = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
    sy = (df["left_shoulder_y"] + df["right_shoulder_y"]) / 2
    hx = (df["left_hip_x"] + df["right_hip_x"]) / 2
    hy = (df["left_hip_y"] + df["right_hip_y"]) / 2
    ay = (df["left_ankle_y"] + df["right_ankle_y"]) / 2
    xs = df[[f"{k}_x" for k in KPTS]].to_numpy()
    ys = df[[f"{k}_y" for k in KPTS]].to_numpy()
    width, height = xs.max(1) - xs.min(1), ys.max(1) - ys.min(1)
    return pd.DataFrame({
        "torso_angle": np.degrees(np.arctan2(np.abs(sx - hx), np.abs(hy - sy))).round(1),   # 0＝直立、90＝躺平
        "aspect": (width / np.maximum(height, 1)).round(3),                                 # 寬÷高
        "hip_height": ((ay - hy) / np.maximum(height, 1)).round(3),                         # 髖離腳多高
    }, index=df.index)


def window_features(df, seconds=1.0, fps=10):
    """每一段序列，只看「過去 1 秒」：姿勢變了多少、髖往下掉多快。"""
    f = frame_features(df).join(df[["seq_id"]])
    n = int(seconds * fps)
    g = f.groupby("seq_id")
    out = f[["torso_angle", "aspect", "hip_height"]].copy()
    out["angle_change"] = g["torso_angle"].diff(n).abs().fillna(0)
    out["hip_drop"] = (-g["hip_height"].diff(n)).fillna(0)
    out["angle_max_1s"] = g["torso_angle"].transform(lambda s: s.rolling(n, min_periods=1).max())
    return out.round(3)


def sequence_table(df):
    """一段序列一列：最後一格的姿勢＋整段最大的變化；答案是這段有沒有要注意的事。"""
    w = window_features(df).join(df[["seq_id", "subject_id", "alert"]])
    agg = w.groupby("seq_id").agg(subject_id=("subject_id", "first"), torso_angle=("torso_angle", "last"),
                                  aspect=("aspect", "last"), hip_height=("hip_height", "last"),
                                  angle_change=("angle_change", "max"), hip_drop=("hip_drop", "max"),
                                  angle_max_1s=("angle_max_1s", "max"), alert=("alert", "max"))
    return agg[agg["alert"] != ""].assign(alert=lambda t: t["alert"].astype(int))


POSE_FEATURES = ["torso_angle", "aspect", "hip_height", "angle_change", "hip_drop", "angle_max_1s"]


def split_by_subject(table, test_size=0.25):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
                  .split(table, groups=table["subject_id"]))
    return table.iloc[tr], table.iloc[te]
#@v31


def load_stream(name="detections.csv"):
    df = pd.read_csv(DATA / "stream" / name)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.drop_duplicates().set_index("ts").sort_index()


def per_minute(stream):
    """不規則的偵測串流 → 每分鐘一格：幾筆偵測、最高信心值。"""
    out = stream.resample("1min").agg({"confidence": ["count", "max"]})
    out.columns = ["detections", "max_conf"]
    return out
#@v32


def add_history(minutes):
    """每一格只用「這一分鐘以前」的資料：過去 10 分鐘的偵測數、過去 1 小時的平均。"""
    m = minutes.copy()
    m["past_10min"] = m["detections"].rolling("10min", closed="left").sum().fillna(0)
    m["past_1h_mean"] = m["detections"].rolling("1h", closed="left").mean().fillna(0).round(3)
    return m
#@v33


def load_sensor():
    s = pd.read_csv(DATA / "stream" / "sensor.csv")
    s["ts"] = pd.to_datetime(s["ts"])
    return s.sort_values("ts")


def align(minutes, sensor, tolerance="2min"):
    """每一格對上「這一分鐘以前、最近的一筆」感測值；超過容許時間就當作沒有。"""
    left = minutes.reset_index().rename(columns={"ts": "minute"})
    return pd.merge_asof(left, sensor[["ts", "value"]], left_on="minute", right_on="ts",
                         direction="backward", tolerance=pd.Timedelta(tolerance)).drop(columns="ts")
#@v34


def psi(expected, actual, bins=10):
    """兩組數字分布差多少（Population Stability Index）：0.1 以下差不多，0.25 以上明顯變了。"""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
#@v35


def hourly_counts():
    """每小時的偵測數（告警量的來源）＋時間特徵。"""
    h = load_stream()["confidence"].resample("1h").count().rename("count").to_frame()
    h["hour"] = h.index.hour
    h["weekday"] = h.index.dayofweek
    h["last_hour"] = h["count"].shift(1)
    h["same_hour_yesterday"] = h["count"].shift(24)
    return h.dropna()


FORECAST_FEATURES = ["hour", "weekday", "last_hour", "same_hour_yesterday"]
#@v37


def pose_cv():
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
#@v39


def weighted_recall(train, test, class_weight):
    model = HistGradientBoostingClassifier(class_weight=class_weight, random_state=0)
    model.fit(train[POSE_FEATURES], train["alert"])
    pred = model.predict(test[POSE_FEATURES])
    return {"class_weight": str(class_weight), "precision": round(precision_score(test["alert"], pred, zero_division=0), 3),
            "recall": round(recall_score(test["alert"], pred), 3), "alerts": int(pred.sum())}
#@all


def main():
#@main v25
    index, roi = load_roi()
    img = read_frame(index["file"].iloc[0])
    print(f"== {LINE}：{len(index)} 張固定攝影機畫面；ROI {roi['name']}（{roi['event_type']}）==")
    print(f"cv2.imread → shape {img.shape}、dtype {img.dtype}（BGR）")
    print(f"ROI 座標 x {roi['x1']}–{roi['x2']}、y {roi['y1']}–{roi['y2']} → 裁出 {crop_roi(img, roi).shape}")
    print("\n== 白天／夜間、ROI 正常／要注意，各幾張；ROI 的平均亮度 ==")
    index["roi_gray"] = [cv2.cvtColor(crop_roi(read_frame(f), roi), cv2.COLOR_BGR2GRAY).mean() for f in index["file"]]
    print(index.groupby(["night", "label"])["roi_gray"].agg(["count", "mean"]).round(1).to_string())
#@main v26 v27
    index, roi = load_roi()
    ref = make_reference(index, roi)
#@main v26
    index["score"] = [roi_score(read_frame(f), roi, ref) for f in index["file"]]
    print(f"== {LINE}：ROI 和參考畫面差多少（最大一塊占 ROI 的比例）==")
    print(index.groupby(["night", "label"])["score"].describe()[["count", "min", "50%", "max"]].round(3).to_string())
    print(f"\n== ROI 被占用的門檻掃一遍；這條線「{roi['alert_when']}」才發事件 ==")
    for t in [0.02, 0.05, 0.1, 0.2]:
        occupied = index["score"] >= t
        pred = (occupied if roi["alert_when"] == "occupied" else ~occupied).astype(int)
        print(f"門檻 {t}：抓到 {int(((pred == 1) & (index['label'] == 1)).sum())}/{int(index['label'].sum())}、"
              f"誤報 {int(((pred == 1) & (index['label'] == 0)).sum())}")
#@main v27
    cam = Camera(index)
    print(f"== {LINE}：{'EZCam ' + cam.url if cam.url else 'Mock 模式（沒有設 EZCAM_URL）'} ==")
    rows = []
    for _ in range(6):
        name, img = cam.capture()
        score = roi_score(img, roi, ref)
        rows.append({"frame": name, "shape": str(img.shape), "score": round(score, 3),
                     "event": roi["event_type"] if decide(score, roi) else "—"})
    print(pd.DataFrame(rows).to_string(index=False))
#@main v28
    index, roi = load_roi()
    gray = cv2.cvtColor(read_frame(index["file"].iloc[0]), cv2.COLOR_BGR2GRAY)
    x = torch.tensor(gray, dtype=torch.float32)[None, None] / 255
    conv = edge_kernel()
    with torch.no_grad():
        edges = conv(x)[0, 0].numpy()
    ref = cv2.filter2D(gray.astype(np.float32) / 255, -1, conv.weight[0, 0].detach().numpy(), borderType=cv2.BORDER_CONSTANT)
    print(f"== {LINE}：nn.Conv2d 放一個 3×3 垂直邊濾波器 ==")
    print(f"輸入 {tuple(x.shape)} → 輸出 {tuple(conv(x).shape)}；參數 {sum(p.numel() for p in conv.parameters())} 個")
    print(f"和 cv2.filter2D 比：最大差 {np.abs(edges - ref).max():.6f}")
    col = np.abs(edges).mean(axis=0)[2:-2]          # 去掉最左右兩格：補 0 的邊界一定有假的邊
    print("邊最強的 4 個 x 座標：", sorted((np.argsort(col)[-4:] + 2).tolist()), f"（ROI 邊界 x={roi['x1']}、{roi['x2']}）")
    layer = nn.Conv2d(3, 16, kernel_size=3)
    print(f"\n一層 Conv2d(3, 16, 3)：權重 {tuple(layer.weight.shape)}，共 {sum(p.numel() for p in layer.parameters())} 個參數，這些數字是訓練學出來的")
#@main v29
    model = load_pose_model()
    photo = STORY / "data" / "photos" / ("factory-lifting.jpg" if LINE == "factory" else "care-corridor-fall.jpg")
    xy, conf = keypoints(model, photo)
    print(f"== {LINE}：YOLO-pose 看 {photo.name} ==")
    print(f"偵測到 {xy.shape[0]} 個人；座標 shape {xy.shape}、信心值 shape {conf.shape}")
    person = pd.DataFrame({"keypoint": KPTS, "x": xy[0, :, 0].astype(float).round(1),
                           "y": xy[0, :, 1].astype(float).round(1), "conf": conf[0].astype(float).round(2)})
    print(person.iloc[[0, 5, 6, 11, 12, 15, 16]].to_string(index=False))
    print(f"信心值低於 0.3 的點：{int((conf[0] < 0.3).sum())} 個")
#@main v30
    poses = load_poses()
    print(f"== {LINE}：骨架資料 {poses['seq_id'].nunique()} 段、{len(poses)} 格（10 fps，每段 4 秒）==")
    table = sequence_table(poses)
    print(f"要注意的段落：{int(table['alert'].sum())} / {len(table)}")
    print(table.groupby("alert")[POSE_FEATURES].mean().round(3).to_string())
    train, test = split_by_subject(table)
    model = HistGradientBoostingClassifier(random_state=0).fit(train[POSE_FEATURES], train["alert"])
    pred = model.predict(test[POSE_FEATURES])
    print(f"\n依人分組切（test 是沒看過的 {test['subject_id'].nunique()} 個人）：precision "
          f"{precision_score(test['alert'], pred):.3f}、recall {recall_score(test['alert'], pred):.3f}")
#@main v31
    stream = load_stream()
    raw = pd.read_csv(DATA / "stream" / "detections.csv")
    print(f"== {LINE}：偵測器原始串流 {len(raw)} 筆，去掉重複剩 {len(stream)} 筆 ==")
    print(f"相鄰兩筆的時間差：中位數 {stream.index.to_series().diff().median()}、最大 {stream.index.to_series().diff().max()}")
    minutes = per_minute(stream)
    print(f"\n== resample('1min')：{len(minutes)} 格 ==")
    print(minutes.iloc[480:486].to_string())
    print(f"\n沒有偵測的分鐘：{int((minutes['detections'] == 0).sum())}（max_conf 是 NaN）")
#@main v32 v33
    minutes = add_history(per_minute(load_stream()))
#@main v32
    print(f"== {LINE}：只看過去（rolling closed='left'）==")
    print(minutes.iloc[480:486].to_string())
    peek = minutes["detections"].rolling("10min").sum()
    print(f"\n不加 closed='left'：第 481 格的 past_10min 是 {peek.iloc[481]:.0f}，"
          f"加了是 {minutes['past_10min'].iloc[481]:.0f}（差的就是「這一分鐘自己」）")
#@main v33
    sensor = load_sensor()
    table = align(minutes, sensor)
    print(f"== {LINE}：merge_asof 對上 {sensor['source'].iloc[0]}（{len(sensor)} 筆）==")
    print(table.iloc[480:486].to_string(index=False))
    print(f"\n對不到（超過 2 分鐘沒有感測值）：{int(table['value'].isna().sum())} 格")
#@main v34
    old = load_stream()
    new = load_stream("detections-new.csv")
    print(f"== {LINE}：上線後 2 週，每支攝影機每週的信心值 PSI（和上線前比）==")
    rows = []
    for cam in sorted(new["source"].unique()):
        base = old[old["source"] == cam]["confidence"].to_numpy()
        for week, part in new[new["source"] == cam].groupby(pd.Grouper(freq="W-SUN")):
            if len(part) < 100:   # 不滿一週的尾巴不算
                continue
            rows.append({"camera": cam, "week_end": week.date(), "n": len(part),
                         "median_conf": round(part["confidence"].median(), 3), "PSI": round(psi(base, part["confidence"].to_numpy()), 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nPSI > 0.25 → 先查攝影機有沒有被移動，再用最近有標籤的資料重新評估，不要直接調門檻")
#@main v35 v36
    hours = hourly_counts()
    split = int(len(hours) * 0.7)
    train, test = hours.iloc[:split], hours.iloc[split:]
#@main v35
    model = LinearRegression().fit(train[FORECAST_FEATURES], train["count"])
    pred = model.predict(test[FORECAST_FEATURES])
    print(f"== {LINE}：預測下一個小時的偵測數（前 70% 學、後 30% 考）==")
    print(f"baseline（用上一小時當預測）MAE {mean_absolute_error(test['count'], test['last_hour']):.2f}")
    print(f"baseline（用訓練期平均）    MAE {mean_absolute_error(test['count'], [train['count'].mean()] * len(test)):.2f}")
    print(f"LinearRegression           MAE {mean_absolute_error(test['count'], pred):.2f}")
    print("\n係數：", ", ".join(f"{f} {c:+.2f}" for f, c in zip(FORECAST_FEATURES, model.coef_)))
#@main v36
    tscv = TimeSeriesSplit(n_splits=4)
    print(f"== {LINE}：TimeSeriesSplit（永遠用過去預測未來）==")
    for k, (tr, va) in enumerate(tscv.split(hours), 1):
        m = LinearRegression().fit(hours.iloc[tr][FORECAST_FEATURES], hours.iloc[tr]["count"])
        err = mean_absolute_error(hours.iloc[va]["count"], m.predict(hours.iloc[va][FORECAST_FEATURES]))
        print(f"第 {k} 折：學 {len(tr):>3} 小時（到 {hours.index[tr[-1]]:%m-%d %H}）、考 {len(va)} 小時 → MAE {err:.2f}")
    m = LinearRegression().fit(train[FORECAST_FEATURES], train["count"])
    resid = test["count"] - m.predict(test[FORECAST_FEATURES])
    print("\n== 殘差（實際 − 預測）依小時看：哪些時段老是猜錯 ==")
    print(resid.groupby(test["hour"]).mean().round(2).sort_values().iloc[[0, 1, -2, -1]].to_string())
#@main v37 v38 v39 v40
    table = sequence_table(load_poses())
    train, test = split_by_subject(table)
#@main v37
    sizes, tr_scores, va_scores = learning_curve(
        HistGradientBoostingClassifier(random_state=0), train[POSE_FEATURES], train["alert"],
        groups=train["subject_id"], cv=pose_cv(), train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        scoring="average_precision")
    print(f"== {LINE}：learning_curve（骨架段落，average precision）==")
    print(pd.DataFrame({"訓練段數": sizes, "train": tr_scores.mean(1).round(3), "valid": va_scores.mean(1).round(3),
                        "valid 標準差": va_scores.std(1).round(3)}).to_string(index=False))
#@main v38
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=0),
        {"learning_rate": loguniform(0.01, 0.3), "max_depth": randint(2, 8), "max_leaf_nodes": randint(8, 48)},
        n_iter=12, scoring="average_precision", cv=pose_cv(), random_state=0, n_jobs=1)
    search.fit(train[POSE_FEATURES], train["alert"], groups=train["subject_id"])
    print(f"== {LINE}：RandomizedSearchCV（12 組，隨機抽）==")
    res = pd.DataFrame(search.cv_results_)[["param_learning_rate", "param_max_depth", "param_max_leaf_nodes", "mean_test_score"]]
    print(res.sort_values("mean_test_score", ascending=False).head(4).round(3).to_string(index=False))
    print(f"\n最佳 CV average precision {search.best_score_:.3f}")
#@main v39
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train["alert"])
    print(f"== {LINE}：少數類（要注意的段落占 {train['alert'].mean():.1%}）==")
    print(f"compute_class_weight('balanced')：0 → {weights[0]:.3f}、1 → {weights[1]:.3f}")
    print(pd.DataFrame([weighted_recall(train, test, None), weighted_recall(train, test, "balanced"),
                        weighted_recall(train, test, {0: 1, 1: 10})]).to_string(index=False))
#@main v40
    poses = load_poses()
    coords = poses[[f"{k}_{a}" for k in KPTS for a in "xy"]]
    coords = coords.sub(coords.mean(axis=1), axis=0)
    pca = PCA(n_components=10, random_state=0).fit(coords)
    print(f"== {LINE}：PCA，把 34 個座標壓成幾個主成分 ==")
    print("前 5 個主成分各解釋多少變化：", np.round(pca.explained_variance_ratio_[:5], 3).tolist())
    print(f"前 3 個加起來：{pca.explained_variance_ratio_[:3].sum():.3f}")
    z = pd.DataFrame(pca.transform(coords)[:, :2], columns=["pc1", "pc2"]).join(poses[["posture"]])
    print("\n每種姿勢在前兩個主成分上的平均：")
    print(z[z["posture"] != ""].groupby("posture")[["pc1", "pc2"]].mean().round(1).to_string())
#@all


if __name__ == "__main__":
    main()
