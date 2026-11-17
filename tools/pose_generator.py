"""骨架序列產生器（給 tools/build_story_extra.py 用）：受試者、攝影機、7 種活動，10 fps、每段 4 秒、COCO 17 點。

不要直接執行；build_story_extra.py 會呼叫 build_pose()。全部是程式產生，不含真實人物。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


KPTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]

# ---------------------------------------------------------------- 骨架

CAMERAS = {
    # 名稱: (畫面寬, 高, 俯角壓縮 vy, 基準縮放, 光線雜訊)
    "cam1": dict(desc="寢室，平視", vy=0.95, scale=1.00, noise=2.0, bed=True),
    "cam2": dict(desc="走廊，高角落", vy=0.62, scale=0.80, noise=3.0, bed=False),
    "cam3": dict(desc="交誼廳，側面", vy=0.85, scale=0.90, noise=2.5, bed=False),
}
SITE2_CAMERAS = {
    "s2cam1": dict(desc="二館寢室，天花板", vy=0.45, scale=0.75, noise=4.5, bed=True),
    "s2cam2": dict(desc="二館走廊，夜間", vy=0.55, scale=0.70, noise=6.0, bed=False),
}
W, H = 640, 480
FPS = 10
NFRAMES = 40

# 各段身體比例（以身高 1.0 為單位）
SEG = dict(head=0.13, neck=0.05, torso=0.30, upper_arm=0.17, forearm=0.15,
           thigh=0.24, shin=0.24, shoulder_w=0.22, hip_w=0.16)


def _rot(v, ang):
    """把向量 v 逆時針轉 ang 弧度（畫面座標 y 向下，故用 -ang 的直覺：正角度＝往前傾）。"""
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def body_points(p):
    """由姿勢參數算 17 點（身體座標，單位＝身高，y 向上，hip 中心在原點）。

    p: torso（軀幹與垂直夾角，正＝前傾）、thigh_l/r、shin_l/r（相對垂直）、
       arm_l/r、fore_l/r、facing（+1 面右、-1 面左）、hip_y（髖高）
    """
    f = p["facing"]
    up = np.array([0.0, 1.0])
    pts = {}
    hip = np.array([0.0, p["hip_y"]])
    tdir = _rot(up, -f * p["torso"])
    sh = hip + tdir * SEG["torso"]
    neck = sh + tdir * SEG["neck"]
    head = neck + tdir * SEG["head"] * 0.6
    side = _rot(tdir, -math.pi / 2) * f  # 軀幹的側向（深度方向，畫面上縮小）
    depth = 0.35
    pts["left_shoulder"] = sh + side * SEG["shoulder_w"] / 2 * depth
    pts["right_shoulder"] = sh - side * SEG["shoulder_w"] / 2 * depth
    pts["left_hip"] = hip + side * SEG["hip_w"] / 2 * depth
    pts["right_hip"] = hip - side * SEG["hip_w"] / 2 * depth
    fwd = _rot(tdir, -math.pi / 2 * f)
    pts["nose"] = head + fwd * 0.05 * f * f
    pts["left_eye"] = head + fwd * 0.035 + tdir * 0.02 + side * 0.015
    pts["right_eye"] = head + fwd * 0.035 + tdir * 0.02 - side * 0.015
    pts["left_ear"] = head - fwd * 0.01 + side * 0.03
    pts["right_ear"] = head - fwd * 0.01 - side * 0.03
    for s in ("left", "right"):
        k = s[0]
        down = np.array([0.0, -1.0])
        knee = pts[f"{s}_hip"] + _rot(down, f * p[f"thigh_{k}"]) * SEG["thigh"]
        ankle = knee + _rot(down, f * p[f"shin_{k}"]) * SEG["shin"]
        elbow = pts[f"{s}_shoulder"] + _rot(-tdir, f * p[f"arm_{k}"]) * SEG["upper_arm"]
        wrist = elbow + _rot(-tdir, f * p[f"fore_{k}"]) * SEG["forearm"]
        pts[f"{s}_knee"], pts[f"{s}_ankle"] = knee, ankle
        pts[f"{s}_elbow"], pts[f"{s}_wrist"] = elbow, wrist
    return np.array([pts[k] for k in KPTS])


POSTURES = {
    # 基本姿勢參數（弧度）；hip_y 以身高為單位
    "standing": dict(torso=0.03, thigh_l=0.0, thigh_r=0.0, shin_l=0.0, shin_r=0.0,
                     arm_l=0.1, arm_r=0.1, fore_l=0.1, fore_r=0.1, hip_y=0.48),
    "sitting": dict(torso=0.08, thigh_l=-1.5, thigh_r=-1.45, shin_l=0.05, shin_r=0.0,
                    arm_l=0.4, arm_r=0.4, fore_l=-0.9, fore_r=-0.9, hip_y=0.26),
    "bending": dict(torso=1.35, thigh_l=0.25, thigh_r=0.2, shin_l=-0.05, shin_r=0.0,
                    arm_l=-0.1, arm_r=-0.05, fore_l=0.0, fore_r=0.0, hip_y=0.46),
    "lying": dict(torso=1.57, thigh_l=-1.57, thigh_r=-1.5, shin_l=-1.57, shin_r=-1.45,
                  arm_l=0.3, arm_r=-0.2, fore_l=0.2, fore_r=0.1, hip_y=0.06),
    "floor_sit": dict(torso=0.35, thigh_l=-1.57, thigh_r=-1.45, shin_l=-1.4, shin_r=-1.2,
                      arm_l=-0.5, arm_r=-0.4, fore_l=0.0, fore_r=0.0, hip_y=0.04),
    "squatting": dict(torso=0.45, thigh_l=-1.25, thigh_r=-1.2, shin_l=0.55, shin_r=0.5,
                      arm_l=-0.3, arm_r=-0.3, fore_l=0.0, fore_r=0.0, hip_y=0.22),
}

# 活動 → (每格姿勢的函式)；t 介於 0..1
ACTIVITIES = ["walk", "stand", "sit", "sit_down", "bend_pick", "fall", "lie_bed"]


def lerp(a, b, w):
    return {k: a[k] + (b[k] - a[k]) * w for k in a}


def posture_seq(activity, rng):
    """回傳 (每格參數 list, 每格 posture 標籤 list, 每格 x 位移 list, fall_frame or -1)。"""
    n = NFRAMES
    params, labels, xs = [], [], []
    fall_at = -1
    speed = rng.uniform(0.015, 0.03) if activity in ("walk", "fall") else 0.0
    jit = {k: float(rng.normal(0, 0.15)) for k in POSTURES["standing"] if k != "hip_y"}
    if activity == "fall":
        end_pose = "floor_sit" if rng.random() < 0.3 else "lying"  # 三成是跌坐，不是躺平
        start = int(rng.integers(8, 18))
        dur = int(rng.integers(5, 10))  # 0.5–1.0 秒倒地
        fall_at = start
    if activity == "sit_down":
        start, dur = int(rng.integers(8, 16)), int(rng.integers(3, 6))  # 很快坐下（困難負例）
    if activity == "bend_pick":
        start, dur = int(rng.integers(6, 14)), int(rng.integers(4, 7))
        hold = int(rng.integers(6, 14))
    for i in range(n):
        ph = 2 * math.pi * i / 10
        st = dict(POSTURES["standing"])
        if activity == "walk":
            sw = 0.35 * math.sin(ph)
            st.update(thigh_l=sw, thigh_r=-sw, shin_l=max(0, -sw) * 0.6, shin_r=max(0, sw) * 0.6,
                      arm_l=-sw * 0.6, arm_r=sw * 0.6)
            p, lab = st, "standing"
        elif activity == "stand":
            p, lab = st, "standing"
        elif activity == "sit":
            p, lab = dict(POSTURES["sitting"]), "sitting"
        elif activity == "lie_bed":
            p, lab = dict(POSTURES["lying"]), "lying"
        elif activity == "sit_down":
            w = min(1.0, max(0.0, (i - start) / dur))
            p = lerp(st, POSTURES["sitting"], w)
            lab = "sitting" if w >= 0.5 else "standing"
        elif activity == "bend_pick":
            if i < start:
                w = 0.0
            elif i < start + dur:
                w = (i - start) / dur
            elif i < start + dur + hold:
                w = 1.0
            elif i < start + 2 * dur + hold:
                w = 1 - (i - start - dur - hold) / dur
            else:
                w = 0.0
            target = POSTURES["bending"] if rng.random() < 0.6 else POSTURES["squatting"]
            p = lerp(st, target, w)
            lab = "bending" if w >= 0.5 else "standing"
        elif activity == "fall":
            sw = 0.35 * math.sin(ph)
            walkp = dict(st, thigh_l=sw, thigh_r=-sw, arm_l=-sw * 0.6, arm_r=sw * 0.6)
            if i < start:
                p, lab = walkp, "standing"
            else:
                w = min(1.0, (i - start) / dur)
                w = w * w * (3 - 2 * w)  # 先慢後快再停
                p = lerp(walkp, POSTURES[end_pose], w)
                lab = ("lying" if end_pose == "lying" else "sitting") if w >= 0.5 else "standing"
        else:
            raise ValueError(activity)
        moving = activity == "walk" or (activity == "fall" and i < start)
        xs.append(speed * i if moving else (speed * start if activity == "fall" else 0.0))
        p = {k: v + (jit.get(k, 0.0) + float(rng.normal(0, 0.04)) if k != "hip_y" else 0.0) for k, v in p.items()}
        params.append(p)
        labels.append(lab)
    return params, labels, xs, fall_at


def render(params, xs, cam, subj, rng):
    """把身體座標投影到 640×480 畫面，加雜訊、遮擋、信心值。"""
    height_px = 300 * cam["scale"] * subj["height"]
    x0 = (rng.uniform(0.2, 0.35) if subj["facing"] > 0 else rng.uniform(0.65, 0.8)) * W
    y0 = rng.uniform(0.80, 0.92) * H  # 腳底所在的畫面高度
    facing = subj["facing"]
    out = []
    occl_lower = cam["bed"] and rng.random() < 0.35  # 寢室：床擋住下半身
    for p, dx in zip(params, xs):
        q = dict(p, facing=facing)
        pts = body_points(q)
        xy = np.empty_like(pts)
        xy[:, 0] = x0 + (pts[:, 0] + dx * facing) * height_px
        xy[:, 1] = y0 - pts[:, 1] * height_px * cam["vy"]
        xy += rng.normal(0, cam["noise"], xy.shape)
        conf = np.clip(rng.normal(0.88, 0.07, len(KPTS)), 0.3, 0.99)
        if occl_lower:
            idx = [13, 14, 15, 16]
            conf[idx] = rng.uniform(0.05, 0.35, len(idx))
            xy[idx] += rng.normal(0, 25, (len(idx), 2))
        # 背對鏡頭側的耳朵、眼睛較不可信
        far = [3, 1] if facing > 0 else [4, 2]
        conf[far] *= 0.6
        if rng.random() < 0.03:  # 偵測器左右搞反
            sw = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            xy, conf = xy[sw], conf[sw]
        xy[:, 0] = np.clip(xy[:, 0], 0, W - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, H - 1)
        lost = rng.random(len(KPTS)) < 0.04  # 偵測器沒抓到的點：座標回 0、信心很低（YOLO 的慣例）
        xy[lost] = 0.0
        conf[lost] = rng.uniform(0.0, 0.1, lost.sum())
        out.append((xy, conf))
    return out


def build_pose(cameras, n_subjects, reps, prefix, seed, label_frac=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    seq_no = 0
    for s in range(1, n_subjects + 1):
        subj = dict(id=f"{prefix}{s:02d}", height=rng.uniform(0.85, 1.12),
                    facing=1 if rng.random() < 0.5 else -1)
        for act in ACTIVITIES:
            for r in range(reps):
                cam_name = list(cameras)[int(rng.integers(len(cameras)))]
                cam = cameras[cam_name]
                if act == "lie_bed" and not cam["bed"]:
                    cam_name = [c for c in cameras if cameras[c]["bed"]][0]
                    cam = cameras[cam_name]
                params, labels, xs, fall_at = posture_seq(act, rng)
                frames = render(params, xs, cam, subj, rng)
                seq_no += 1
                seq_id = f"{prefix}Q{seq_no:04d}"
                labeled = rng.random() < label_frac
                # 約 2% 的序列，人工把活動標錯（真實標註也會錯）
                act_label = act
                if labeled and rng.random() < 0.02:
                    act_label = {"fall": "lie_bed", "lie_bed": "fall", "bend_pick": "fall",
                                 "sit_down": "sit"}.get(act, act)
                for i, ((xy, conf), lab) in enumerate(zip(frames, labels)):
                    row = dict(seq_id=seq_id, subject_id=subj["id"], camera_id=cam_name, frame=i,
                               t=round(i / FPS, 1),
                               activity=act_label if labeled else "",
                               posture=lab if labeled else "",
                               is_fall_seq=int(act_label == "fall") if labeled else "",
                               fall_frame=fall_at if (labeled and act_label == "fall") else (-1 if labeled else ""))
                    for k, name in enumerate(KPTS):
                        row[f"{name}_x"] = round(float(xy[k, 0]), 1)
                        row[f"{name}_y"] = round(float(xy[k, 1]), 1)
                        row[f"{name}_c"] = round(float(conf[k]), 3)
                    rows.append(row)
    return pd.DataFrame(rows)
