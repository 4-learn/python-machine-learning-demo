"""偵測器：看照片，把「要注意」的畫面變成事件送進平台（ML 課進度存檔 v16）。

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

LINE = os.getenv("LINE", "factory")
HERE = Path(__file__).resolve().parent
STORY = HERE.parent
PROFILE = json.loads((STORY / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
CROPS = STORY / "data" / LINE / "crops"
SCENES = STORY / "data" / LINE / "scenes"
NIGHT_CAMERA = {"factory": "camera_04", "care": "camera_14"}[LINE]
EVENT_TYPE = {"factory": "no_helmet", "care": "fall_detected"}[LINE]


def load_index():
    """小圖清單：檔名、哪一支攝影機、ok（沒事）或 alert（要注意）。"""
    return pd.read_csv(CROPS / "index.csv")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def main():
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


if __name__ == "__main__":
    main()
