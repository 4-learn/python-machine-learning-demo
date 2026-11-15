"""進階：攝影機 ROI、骨架、串流資料、判斷器進階（ML 課進度存檔 v32，AI 新秀）。

執行（在這個資料夾裡）：
    LINE=factory python extra.py     # 工廠線：通道被占用、彎腰搬重物
    LINE=care python extra.py        # 健康線：夜間離床、跌倒
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import requests
import torch
from torch import nn
from ultralytics import YOLO
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

LINE = os.getenv("LINE", "factory")
HERE = Path(__file__).resolve().parent
STORY = HERE.parent
PROFILE = json.loads((STORY / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
DATA = STORY / "data" / LINE


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


def edge_kernel():
    """一個 3×3 的「找垂直邊」濾波器，放進 nn.Conv2d。"""
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])
    return conv


KPTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]


def load_pose_model():
    return YOLO(str(STORY.parent / "models" / "yolo11n-pose.pt"))


def keypoints(model, image_path):
    """一張照片 → (人數, 17, 2) 的座標與 (人數, 17) 的信心值。"""
    result = model.predict(str(image_path), imgsz=640, device="cpu", verbose=False)[0]
    return result.keypoints.xy.numpy(), result.keypoints.conf.numpy()


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


def load_stream(name="detections.csv"):
    df = pd.read_csv(DATA / "stream" / name)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.drop_duplicates().set_index("ts").sort_index()


def per_minute(stream):
    """不規則的偵測串流 → 每分鐘一格：幾筆偵測、最高信心值。"""
    out = stream.resample("1min").agg({"confidence": ["count", "max"]})
    out.columns = ["detections", "max_conf"]
    return out


def add_history(minutes):
    """每一格只用「這一分鐘以前」的資料：過去 10 分鐘的偵測數、過去 1 小時的平均。"""
    m = minutes.copy()
    m["past_10min"] = m["detections"].rolling("10min", closed="left").sum().fillna(0)
    m["past_1h_mean"] = m["detections"].rolling("1h", closed="left").mean().fillna(0).round(3)
    return m


def main():
    minutes = add_history(per_minute(load_stream()))
    print(f"== {LINE}：只看過去（rolling closed='left'）==")
    print(minutes.iloc[480:486].to_string())
    peek = minutes["detections"].rolling("10min").sum()
    print(f"\n不加 closed='left'：第 481 格的 past_10min 是 {peek.iloc[481]:.0f}，"
          f"加了是 {minutes['past_10min'].iloc[481]:.0f}（差的就是「這一分鐘自己」）")


if __name__ == "__main__":
    main()
