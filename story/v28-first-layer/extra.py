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
from torchvision.models import ResNet18_Weights, resnet18

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


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)   # 26 節之後常用的「找垂直邊」濾波器


def first_layer(pretrained=True):
    """ResNet18 的第一層：64 個 7×7 的濾波器，每個看 R、G、B 三個顏色。pretrained=False 是沒訓練過的對照組。"""
    torch.manual_seed(0)
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return resnet18(weights=weights).eval().conv1


def kernel_similarity(kernel, pattern):
    """學出來的濾波器（三色平均，7×7）和放大到 7×7 的手寫 pattern 比像不像（相關係數，-1～1）。
    幾乎全是 0 的濾波器（沒在用）回傳 0。"""
    k = kernel.detach().mean(0).numpy()
    if np.abs(k).max() < 1e-3:
        return 0.0
    big = cv2.resize(pattern, k.shape[::-1], interpolation=cv2.INTER_LINEAR)
    return float(np.corrcoef(k.ravel(), big.ravel())[0, 1])


def main():
    index, roi = load_roi()
    conv = first_layer()
    print(f"== {LINE}：ResNet18 第一層（ImageNet 預訓練）==")
    print(f"權重 shape {tuple(conv.weight.shape)}：{conv.out_channels} 個濾波器 × {conv.in_channels} 色 × "
          f"{conv.kernel_size[0]}×{conv.kernel_size[1]}；stride {conv.stride[0]}、padding {conv.padding[0]}")
    print(f"參數 {sum(p.numel() for p in conv.parameters()):,} 個，全部是訓練學出來的，沒有人手寫")
    sims = np.array([kernel_similarity(conv.weight[i], SOBEL_X) for i in range(conv.out_channels)])
    dead = int(sum(conv.weight[i].abs().max() < 1e-3 for i in range(conv.out_channels)))
    print(f"\n== 學出來的濾波器，哪幾個最像手寫的「找垂直邊」（Sobel x，比 7×7）==")
    print(f"（{dead} 個濾波器幾乎全是 0，訓練後沒在用，不列入比較）")
    for i in np.argsort(-np.abs(sims))[:3]:
        print(f"第 {i:>2} 個：相關係數 {sims[i]:+.3f}")
    rand = first_layer(pretrained=False)
    rsims = np.array([kernel_similarity(rand.weight[i], SOBEL_X) for i in range(rand.out_channels)])
    print(f"對照：沒訓練過（亂數）的第一層，最像的只有 {np.abs(rsims).max():.3f}")
    rgb = cv2.cvtColor(read_frame(index["file"].iloc[0]), cv2.COLOR_BGR2RGB)
    x = ResNet18_Weights.IMAGENET1K_V1.transforms()(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        maps = conv(x)[0]
    print(f"\n== 把 {index['file'].iloc[0]} 送進第一層 ==")
    print(f"輸入 {tuple(x.shape)} → 輸出 {tuple(maps.shape)}：64 張「特徵圖」，每張是原圖的一半寬高（stride 2）")
    strength = maps.abs().mean(dim=(1, 2)).numpy()
    print(f"反應最強的 3 個濾波器：{np.argsort(-strength)[:3].tolist()}；最弱的 3 個：{np.argsort(strength)[:3].tolist()}")


if __name__ == "__main__":
    main()
