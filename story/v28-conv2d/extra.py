"""進階：攝影機 ROI、骨架、串流資料、判斷器進階（ML 課進度存檔 v28，AI 新秀）。

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


def main():
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


if __name__ == "__main__":
    main()
