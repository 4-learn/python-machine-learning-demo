"""26｜OpenCV：ROI 有沒有被占用。和空的 ROI 相減 → 二值化 → 開運算 → 找輪廓 → 最大一塊有多大。"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ROI_DIR = ROOT / "story" / "data" / "factory" / "roi"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
index = pd.read_csv(ROI_DIR / "index.csv")
roi = json.loads((ROI_DIR / "roi.json").read_text(encoding="utf-8"))


def read_roi(name):
    img = cv2.imread(str(ROI_DIR / "frames" / name))
    if img is None:
        raise FileNotFoundError(name)
    return img[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]


def gray(crop, blur=True, normalize=True):
    if blur:
        crop = cv2.GaussianBlur(crop, (5, 5), 0)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if normalize:
        g = g * (128 / max(float(np.median(g)), 1.0))   # 把 ROI 的中位數亮度拉到 128
    return cv2.convertScaleAbs(g)


# 這一節用的畫面：000 空通道、001 貨物＋有人走過、002 只有人走過、015 夜間空通道、011 夜間有貨物
FRAMES = ["frame_000.jpg", "frame_001.jpg", "frame_002.jpg", "frame_015.jpg", "frame_011.jpg"]
NOTE = {"frame_000.jpg": "白天、空的", "frame_001.jpg": "白天、貨物＋人", "frame_002.jpg": "白天、只有人走過",
        "frame_015.jpg": "夜間、空的", "frame_011.jpg": "夜間、貨物"}
empty = index[(index["night"] == 0) & (index["label"] == 0)]["file"][:15]   # 白天、ROI 空著的 15 張

print("== 1. 和空著的時候直接相減：夜間整塊都「不一樣」 ==")
ref_raw = np.median(np.stack([gray(read_roi(f), blur=False, normalize=False) for f in empty]), axis=0).astype(np.uint8)
for f in FRAMES:
    diff = cv2.absdiff(gray(read_roi(f), blur=False, normalize=False), ref_raw)
    print(f"{f}（{NOTE[f]:<8}）差超過 15 的像素占 {(diff > 15).mean():.3f}")

print("\n== 2. 先整理畫面：亮度歸一＋模糊，再相減 ==")
ref = np.median(np.stack([gray(read_roi(f)) for f in empty]), axis=0).astype(np.uint8)
for f in FRAMES:
    diff = cv2.absdiff(gray(read_roi(f)), ref)
    print(f"{f}（{NOTE[f]:<8}）差超過 15 的像素占 {(diff > 15).mean():.3f}")

print("\n== 3. 黑白遮罩 → 清掉小白點 → 一塊一塊量 ==")
kernel = np.ones((5, 5), np.uint8)
_, mask = cv2.threshold(cv2.absdiff(gray(read_roi("frame_001.jpg")), ref), 15, 255, cv2.THRESH_BINARY)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in sorted(contours, key=cv2.contourArea, reverse=True):
    x, y, w, h = cv2.boundingRect(c)
    print(f"面積 {cv2.contourArea(c):6.1f}（占 ROI {cv2.contourArea(c) / mask.size:.3f}）；外框 寬 {w}、高 {h}")
cv2.imwrite(str(OUT / "frame_001-mask.png"), mask)

print("\n== 4. 一張畫面一個分數：最大一塊占 ROI 幾成 ==")


def roi_score(name):
    diff = cv2.absdiff(gray(read_roi(name)), ref)
    _, m = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max((cv2.contourArea(c) for c in cs), default=0.0) / m.size


rows = [{"file": f, "畫面": NOTE[f], "label": int(index.set_index("file").loc[f, "label"]),
         "score": round(roi_score(f), 3)} for f in FRAMES]
print(pd.DataFrame(rows).to_string(index=False))
