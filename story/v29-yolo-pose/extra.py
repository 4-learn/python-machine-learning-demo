"""進階：攝影機 ROI、骨架、串流資料、判斷器進階（ML 課進度存檔 v29，AI 新秀）。

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


def main():
    model = load_pose_model()
    photo = STORY / "data" / "photos" / ("factory-lifting.jpg" if LINE == "factory" else "care-corridor-fall.jpg")
    xy, conf = keypoints(model, photo)
    print(f"== {LINE}：YOLO-pose 看 {photo.name} ==")
    print(f"偵測到 {xy.shape[0]} 個人；座標 shape {xy.shape}、信心值 shape {conf.shape}")
    person = pd.DataFrame({"keypoint": KPTS, "x": xy[0, :, 0].astype(float).round(1),
                           "y": xy[0, :, 1].astype(float).round(1), "conf": conf[0].astype(float).round(2)})
    print(person.iloc[[0, 5, 6, 11, 12, 15, 16]].to_string(index=False))
    print(f"信心值低於 0.3 的點：{int((conf[0] < 0.3).sum())} 個")


if __name__ == "__main__":
    main()
