"""進階：攝影機 ROI、骨架、串流資料、判斷器進階（ML 課進度存檔 v25，AI 新秀）。

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


def main():
    index, roi = load_roi()
    img = read_frame(index["file"].iloc[0])
    print(f"== {LINE}：{len(index)} 張固定攝影機畫面；ROI {roi['name']}（{roi['event_type']}）==")
    print(f"cv2.imread → shape {img.shape}、dtype {img.dtype}（BGR）")
    print(f"ROI 座標 x {roi['x1']}–{roi['x2']}、y {roi['y1']}–{roi['y2']} → 裁出 {crop_roi(img, roi).shape}")
    print("\n== 白天／夜間、ROI 正常／要注意，各幾張；ROI 的平均亮度 ==")
    index["roi_gray"] = [cv2.cvtColor(crop_roi(read_frame(f), roi), cv2.COLOR_BGR2GRAY).mean() for f in index["file"]]
    print(index.groupby(["night", "label"])["roi_gray"].agg(["count", "mean"]).round(1).to_string())


if __name__ == "__main__":
    main()
