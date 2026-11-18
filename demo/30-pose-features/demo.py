"""30｜NumPy：17 個關鍵點 → 和畫面大小無關的姿勢特徵 → 「過去 1 秒」的時間窗口 → 依人分組切。"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

warnings.simplefilter("ignore", RuntimeWarning)   # 左右兩點都沒看到時 nanmean 會警告「全是 NaN」，結果就是 NaN
ROOT = Path(__file__).resolve().parents[2]
KPTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]

print("== 1. np.arctan2 + np.degrees：髖→肩這條線歪了幾度 ==")
# 手打 4 個人的肩中點、髖中點（像素，y 向下）
toy = pd.DataFrame({"姿勢": ["站著", "微微前傾", "彎腰", "躺平"],
                    "肩 x": [100, 110, 160, 180], "肩 y": [100, 102, 150, 300],
                    "髖 x": [100, 100, 100, 100], "髖 y": [200, 200, 200, 300]})
dx = np.abs(toy["肩 x"] - toy["髖 x"])
dy = np.abs(toy["髖 y"] - toy["肩 y"])          # y 往下變大，所以是「髖 − 肩」
toy["角度"] = np.degrees(np.arctan2(dx, dy)).round(1)
toy["dx"], toy["dy"] = dx, dy
print(toy[["姿勢", "肩 x", "肩 y", "髖 x", "髖 y", "dx", "dy", "角度"]].to_string(index=False))

print("\n== 2. 13,440 格一次算完：先遮掉沒看到的點，再算特徵 ==")
df = pd.read_csv(ROOT / "story" / "data" / "care" / "pose" / "poses.csv", keep_default_na=False)
xs = df[[f"{k}_x" for k in KPTS]].to_numpy()          # (格數, 17)
ys = df[[f"{k}_y" for k in KPTS]].to_numpy()
cs = df[[f"{k}_c" for k in KPTS]].to_numpy()
print(f"xs {xs.shape}、ys {ys.shape}、cs {cs.shape}")
xs_ok = np.where(cs < 0.3, np.nan, xs)
ys_ok = np.where(cs < 0.3, np.nan, ys)

def features(xs, ys):
    sx, sy = np.nanmean(xs[:, [5, 6]], axis=1), np.nanmean(ys[:, [5, 6]], axis=1)       # 肩中點
    hx, hy = np.nanmean(xs[:, [11, 12]], axis=1), np.nanmean(ys[:, [11, 12]], axis=1)   # 髖中點
    angle = np.degrees(np.arctan2(np.abs(sx - hx), np.abs(hy - sy)))
    width = np.nanmax(xs, axis=1) - np.nanmin(xs, axis=1)
    height = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)
    return pd.DataFrame({"torso_angle": angle, "aspect": width / height})


masked = features(xs_ok, ys_ok)
print(masked.join(df["posture"]).groupby("posture").median().round(3).to_string())

print("\n== 3. groupby('seq_id').diff(10)：和 1 秒前（10 格前）比 ==")
df["torso_angle"] = masked["torso_angle"].round(1)
df["change_1s"] = df.groupby("seq_id")["torso_angle"].diff(10).abs()
fall = df[df["seq_id"] == "SQ0039"]
print(f"SQ0039：{fall['activity'].iloc[0]}，攝影機 {fall['camera_id'].iloc[0]}，第 {fall['fall_frame'].iloc[0]} 格開始倒")
print(fall[["frame", "t", "posture", "torso_angle", "change_1s"]].iloc[::4].to_string(index=False))

print("\n== 4. 每一種活動：過去 1 秒最大變化、最後一格的姿勢 ==")
seq = df.groupby("seq_id").agg(activity=("activity", "first"), last_angle=("torso_angle", "last"),
                               max_change_1s=("change_1s", "max"))
print(seq.groupby("activity")[["last_angle", "max_change_1s"]].median().round(1).to_string())

print("\n== 5. 依人分組切：同一個人不能同時在兩邊 ==")
df["aspect"] = masked["aspect"]
X, y = df[["torso_angle", "aspect", "change_1s"]], df["alert"].astype(int)
tr_r, te_r = train_test_split(np.arange(len(df)), test_size=0.25, random_state=0)
rows = []
tr_g, te_g = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                  .split(X, y, groups=df["subject_id"]))
for name, tr, te in [("隨機切格子", tr_r, te_r), ("依人分組（GroupShuffleSplit）", tr_g, te_g)]:
    both_sides = set(df["subject_id"].iloc[tr]) & set(df["subject_id"].iloc[te])
    model = HistGradientBoostingClassifier(random_state=0).fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[te])
    rows.append({"切法": name, "test 格數": len(te), "兩邊都有的人": len(both_sides),
                 "precision": precision_score(y.iloc[te], pred), "recall": recall_score(y.iloc[te], pred)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f"依人分組的 test：{sorted(df['subject_id'].iloc[te_g].unique())}")
