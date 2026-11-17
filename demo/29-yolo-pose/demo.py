"""29｜ultralytics YOLO-pose：一張照片 → 每個人 17 個關鍵點（座標＋信心值），只留骨架、不留畫面。"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # 不上網檢查更新

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
PHOTOS = ROOT / "story" / "data" / "photos"      # AI 生成的擬真照片，不是真實人物
KPTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]

print("== 1. 載入姿勢模型 ==")
model = YOLO(str(ROOT / "models" / "yolo11n-pose.pt"))
print(f"task：{model.task}；類別：{model.names}；每個人的關鍵點：{model.model.kpt_shape}（17 個點 × x, y, 信心值）")
print(f"授權：{model.ckpt['license']}")

print("\n== 2. predict：Results 多了 keypoints ==")
r = model.predict(str(PHOTOS / "factory-lifting.jpg"), imgsz=640, device="cpu", verbose=False)[0]
print(f"原圖大小（高, 寬）：{r.orig_shape}；人數：{len(r.boxes)}；框信心值：{[round(c, 3) for c in r.boxes.conf.tolist()]}")
print(f"keypoints.xy  ：{tuple(r.keypoints.xy.shape)}（人數, 17 個點, x 與 y）")
print(f"keypoints.conf：{tuple(r.keypoints.conf.shape)}（人數, 17 個點）")
print(f"keypoints.data：{tuple(r.keypoints.data.shape)}（x, y, 信心值放在一起）")

print("\n== 3. 一個人的 17 個點（factory-lifting 第 0 人：彎腰搬箱的人）==")
xy = r.keypoints.xy.numpy().astype(float)
conf = r.keypoints.conf.numpy().astype(float)
person = pd.DataFrame({"keypoint": KPTS, "x": xy[0, :, 0].round(1), "y": xy[0, :, 1].round(1),
                       "conf": conf[0].round(3), "低信心": np.where(conf[0] < 0.3, "<0.3", "")})
person.index.name = "編號"
print(person.to_string())

print("\n== 4. 三張照片、每個人：框信心值與低信心的點 ==")
results = {}
for name in ["factory-lifting.jpg", "care-corridor-fall.jpg", "care-room.jpg"]:
    res = model.predict(str(PHOTOS / name), imgsz=640, device="cpu", verbose=False)[0]
    c = res.keypoints.conf.numpy()
    for i, box_conf in enumerate(res.boxes.conf.tolist()):
        low = [KPTS[k] for k in np.where(c[i] < 0.3)[0]]
        print(f"{name:<23} 第 {i} 人：框 {box_conf:.3f}，點信心值平均 {c[i].mean():.3f}，"
              f"低於 0.3 的點 {len(low)} 個" + (f"（{', '.join(low)}）" if low else ""))
    results[name] = res

print("\n== 5. 從點讀姿勢：鼻子、髖、腳踝的位置（y 越大越下面）==")
ROWS = [("factory-lifting.jpg", 0, "彎腰搬箱"), ("factory-lifting.jpg", 1, "站在工作台前"),
        ("care-corridor-fall.jpg", 0, "倒在走廊"), ("care-corridor-fall.jpg", 1, "走過來的照服員")]
rows = []
for name, i, what in ROWS:
    p = results[name].keypoints.xy.numpy()[i].astype(float)     # 這個人的 (17, 2)
    hip = p[[11, 12]].mean(axis=0)
    ankle = p[[15, 16]].mean(axis=0)
    rows.append({"人": what, "鼻子 x": p[0, 0], "鼻子 y": p[0, 1], "髖 x": hip[0], "髖 y": hip[1],
                 "腳踝 x": ankle[0], "腳踝 y": ankle[1]})
print(pd.DataFrame(rows).round(1).to_string(index=False))

print("\n== 6. xy 是像素，xyn 是除以寬高的 0–1 比例 ==")
k = r.keypoints
h, w = r.orig_shape
print(f"left_hip：xy {k.xy[0, 11].tolist()[0]:.1f}, {k.xy[0, 11].tolist()[1]:.1f}；"
      f"xyn {k.xyn[0, 11].tolist()[0]:.3f}, {k.xyn[0, 11].tolist()[1]:.3f}")
print(f"xy ÷ (寬, 高) 等於 xyn：{np.allclose(k.xy.numpy() / [w, h], k.xyn.numpy())}")

print("\n== 7. conf=0.5：把沒把握的「人」拿掉（care-room.jpg）==")
for c in [0.25, 0.5]:
    b = model.predict(str(PHOTOS / "care-room.jpg"), conf=c, imgsz=640, device="cpu", verbose=False)[0].boxes
    print(f"conf={c:<4} 人數 {len(b)}：框信心值 {[round(v, 3) for v in b.conf.tolist()]}")

print("\n== 8. 畫面裡沒有人（ezcam-sample.jpg）==")
empty = model.predict(str(PHOTOS / "ezcam-sample.jpg"), imgsz=640, device="cpu", verbose=False)[0]
print(f"人數 {len(empty.boxes)}；keypoints.xy {tuple(empty.keypoints.xy.shape)}")
print("要先檢查人數：沒有人時 xy[0] 會 IndexError")

print("\n== 9. 只存骨架，不存畫面 ==")
photo = PHOTOS / "factory-lifting.jpg"
row = {f"{kp}_{a}": round(float(v), 3) for kp, (x, y), c in zip(KPTS, xy[0], conf[0])
       for a, v in zip("xyc", (x, y, c))}
cols = [c for c in pd.read_csv(ROOT / "story" / "data" / "factory" / "pose" / "poses.csv", nrows=1).columns
        if c.endswith(("_x", "_y", "_c"))]
print(f"一張照片：{h} × {w} × 3 = {h * w * 3:,} 個數字（JPEG 檔 {photo.stat().st_size:,} bytes）")
print(f"一個骨架：17 × 3 = {len(row)} 個數字（float32 共 {len(row) * 4} bytes）")
print(f"欄名和 poses.csv 同名同序：{list(row) == cols}")
print(f"前 6 欄：{dict(list(row.items())[:6])}")
