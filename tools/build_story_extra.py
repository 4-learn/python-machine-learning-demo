"""產生 ML 課（2026 v2）25–42 節（AI 新秀）兩條線的資料。

    python tools/build_story_extra.py            # 重建
    python tools/build_story_extra.py --check    # 比對 EXTRA-MANIFEST.json

產物（story/data/{factory,care}/ 底下，兩條線同一種格式）：
  roi/frames/*.jpg ＋ roi/index.csv ＋ roi/roi.json   固定攝影機畫面與 ROI（25–27、42）
      工廠：通道 ROI 被貨物占用＝aisle_blocked；健康：床 ROI 裡沒有人＝bed_exit（夜間離床）
  pose/poses.csv          骨架序列（29–30、40、42），沿用國北護課的骨架產生器
      工廠：alert＝彎腰搬重物（bend_pick 且姿勢 bending）；健康：alert＝跌倒（fall）
  pose/poses-site2.csv    新地點的骨架（攝影機更高、更暗），只有 30% 有標籤（34）
  stream/detections.csv   偵測器原始輸出：不規則時間、每格一筆（31–33）
  stream/sensor.csv       第二個來源：工廠＝通道照度計每 10 秒；健康＝床墊壓力每分鐘（33）
  stream/detections-new.csv  上線後 2 週，第 2 週某支攝影機換位置（34）
全部是程式合成，不含真實人物。
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260125


def load_pose_module():
    """骨架產生器（tools/pose_generator.py）。"""
    spec = importlib.util.spec_from_file_location("pose_gen", ROOT / "tools" / "pose_generator.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- ROI 畫面
def roi_frames(line, out, rng):
    root = out / "roi" / "frames"; root.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240
    if line == "factory":
        roi = {"name": "aisle_01", "x1": 110, "y1": 90, "x2": 210, "y2": 240, "event_type": "aisle_blocked", "alert_when": "occupied"}
    else:
        roi = {"name": "room_301", "x1": 60, "y1": 120, "x2": 260, "y2": 200, "event_type": "bed_exit", "alert_when": "empty"}
    rows = []
    for i in range(160):
        night = i % 4 == 3
        base = (150, 148, 140) if line == "factory" else (205, 195, 175)
        img = Image.new("RGB", (W, H), base); d = ImageDraw.Draw(img)
        if line == "factory":
            d.rectangle([0, 150, W, H], fill=(120, 118, 112))                      # 地板
            d.rectangle([roi["x1"], roi["y1"], roi["x2"], roi["y2"]], fill=(170, 170, 160))   # 通道
            d.line([roi["x1"], roi["y1"], roi["x1"], H], fill=(230, 200, 40), width=4)        # 黃線
            d.line([roi["x2"], roi["y1"], roi["x2"], H], fill=(230, 200, 40), width=4)
            for x in (20, 250):                                                   # 兩側貨架
                d.rectangle([x, 40, x + 50, 200], fill=(90, 80, 70))
            blocked = rng.random() < 0.4
            if blocked:                                                           # 通道上的貨物
                bx = int(rng.integers(roi["x1"] + 5, roi["x2"] - 45)); by = int(rng.integers(140, 200))
                d.rectangle([bx, by, bx + int(rng.integers(30, 45)), by + int(rng.integers(25, 40))], fill=(170, 120, 60))
            if rng.random() < 0.3:                                                # 人走過（不算占用）
                px = int(rng.integers(roi["x1"] + 10, roi["x2"] - 10))
                d.rectangle([px - 5, 120, px + 5, 175], fill=(60, 90, 150)); d.ellipse([px - 6, 108, px + 6, 120], fill=(220, 180, 150))
            label = int(blocked)
        else:
            d.rectangle([0, 170, W, H], fill=(170, 160, 140))
            d.rectangle([roi["x1"], roi["y1"], roi["x2"], roi["y2"]], fill=(235, 235, 240))     # 床
            d.rectangle([roi["x1"], roi["y1"] - 10, roi["x1"] + 12, roi["y2"]], fill=(130, 100, 70))  # 床頭
            in_bed = rng.random() < 0.6
            if in_bed:
                y = int(rng.integers(roi["y1"] + 15, roi["y2"] - 25))
                d.ellipse([roi["x1"] + 18, y, roi["x1"] + 42, y + 22], fill=(220, 180, 150))
                d.rectangle([roi["x1"] + 45, y - 4, roi["x2"] - 20, y + 28], fill=(180, 190, 220))
            else:
                d.rectangle([roi["x1"] + 45, roi["y1"] + 20, roi["x2"] - 20, roi["y1"] + 55], fill=(180, 190, 220))  # 掀開的被子
                if rng.random() < 0.5:                                            # 人站在床邊
                    px = int(rng.integers(20, 50)); d.rectangle([px, 120, px + 14, 200], fill=(150, 120, 160))
                    d.ellipse([px - 1, 104, px + 15, 120], fill=(220, 180, 150))
            label = int(not in_bed)
        a = np.asarray(img, np.float32) * (0.45 if night else rng.uniform(0.9, 1.1))
        a += rng.normal(0, 8 if night else 3, a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        if rng.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(0.8))
        name = f"frame_{i:03d}.jpg"; img.save(root / name, quality=90)
        rows.append({"file": name, "night": int(night), "label": label,
                     "taken_at": str(pd.Timestamp("2026-01-19 08:00") + pd.Timedelta(minutes=7 * i))})
    pd.DataFrame(rows).to_csv(out / "roi" / "index.csv", index=False)
    (out / "roi" / "roi.json").write_text(json.dumps(roi, ensure_ascii=False, indent=1))
    return len(rows)


# ---------------------------------------------------------------- 骨架
def poses(line, out, gen):
    p = out / "pose"; p.mkdir(parents=True, exist_ok=True)
    seed = 1 if line == "care" else 3
    main = gen.build_pose(gen.CAMERAS, 24, 2, "S", seed=seed)
    site2 = gen.build_pose(gen.SITE2_CAMERAS, 10, 2, "T", seed=seed + 1, label_frac=0.3)
    for df in (main, site2):
        if line == "factory":
            df["alert"] = np.where(df["activity"] == "", "",
                                   ((df["activity"] == "bend_pick") & (df["posture"] == "bending")).astype(int).astype(str))
        else:
            df["alert"] = np.where(df["activity"] == "", "", (df["is_fall_seq"].astype(str) == "1").astype(int).astype(str))
    main.to_csv(p / "poses.csv", index=False); site2.to_csv(p / "poses-site2.csv", index=False)
    return {"seqs": int(main.seq_id.nunique()), "site2_seqs": int(site2.seq_id.nunique())}


# ---------------------------------------------------------------- 串流
def stream(line, out, rng):
    s = out / "stream"; s.mkdir(parents=True, exist_ok=True)
    cams = ["camera_01", "camera_02", "camera_03"] if line == "factory" else ["camera_11", "camera_12", "camera_13"]
    etype = "no_helmet" if line == "factory" else "fall_detected"

    def make(start, days, shift_cam=None, shift_day=None):
        rows = []
        t = pd.Timestamp(start)
        end = t + pd.Timedelta(days=days)
        while t < end:
            hour = t.hour
            busy = (8 <= hour < 18) if line == "factory" else (hour >= 21 or hour < 7)
            rate = 0.9 if busy else 0.25                         # 每分鐘幾筆偵測
            t += pd.Timedelta(seconds=float(rng.exponential(60 / rate)))
            cam = cams[rng.integers(len(cams))]
            conf = float(np.clip(rng.beta(5, 3), 0.05, 0.99))
            if shift_cam and cam == shift_cam and t >= pd.Timestamp(shift_day):
                conf = float(np.clip(conf * 0.7, 0.05, 0.99))    # 換位置後信心整體偏低
            rows.append({"ts": t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "source": cam, "event_type": etype,
                         "confidence": round(conf, 3), "boxes": int(rng.integers(1, 4))})
        df = pd.DataFrame(rows)
        dup = df.sample(frac=0.03, random_state=int(rng.integers(1e6)))   # 重複上傳
        return pd.concat([df, dup]).sort_values("ts", kind="stable").reset_index(drop=True)

    det = make("2026-01-05", 7); det.to_csv(s / "detections.csv", index=False)
    new = make("2026-01-19", 14, shift_cam=cams[1], shift_day="2026-01-26"); new.to_csv(s / "detections-new.csv", index=False)
    # 第二個來源
    if line == "factory":
        idx = pd.date_range("2026-01-05", "2026-01-12", freq="10s", inclusive="left")
        lux = np.asarray(600 + 150 * np.sin(2 * np.pi * (idx.hour + idx.minute / 60 - 6) / 24)) + rng.normal(0, 20, len(idx))
        lux[np.asarray((idx.hour >= 19) | (idx.hour < 6))] -= 380
        sensor = pd.DataFrame({"ts": idx.strftime("%Y-%m-%d %H:%M:%S"), "source": "lux_aisle_01", "value": np.round(np.clip(lux, 20, None))})
    else:
        idx = pd.date_range("2026-01-05", "2026-01-12", freq="1min", inclusive="left")
        in_bed = np.asarray((idx.hour >= 21) | (idx.hour < 7))
        kpa = np.where(in_bed, rng.normal(4.2, 0.3, len(idx)), rng.normal(0.4, 0.2, len(idx)))
        out_bed = in_bed & (rng.random(len(idx)) < 0.04)
        kpa[out_bed] = rng.normal(0.3, 0.2, out_bed.sum())
        sensor = pd.DataFrame({"ts": idx.strftime("%Y-%m-%d %H:%M:%S"), "source": "bed_301", "value": np.round(np.clip(kpa, 0, None), 2)})
    sensor = sensor.drop(sensor.sample(frac=0.02, random_state=1).index)       # 掉資料
    sensor.to_csv(s / "sensor.csv", index=False)
    return {"detections": len(det), "new": len(new), "sensor": len(sensor)}


def build(base: Path) -> dict:
    gen = load_pose_module()
    stats = {}
    for k, line in enumerate(["factory", "care"]):
        out = base / line
        stats[line] = {"roi": roi_frames(line, out, np.random.default_rng(SEED + k)),
                       "pose": poses(line, out, gen),
                       "stream": stream(line, out, np.random.default_rng(SEED + 10 + k))}
    return stats


def manifest(base: Path) -> dict:
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.rglob("*")) if p.is_file() and any(f"/{d}/" in str(p) for d in ("roi", "pose", "stream"))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    target = ROOT / "story" / "data"
    if a.check:
        with tempfile.TemporaryDirectory() as t:
            build(Path(t)); got = manifest(Path(t))
        want = json.loads((target / "EXTRA-MANIFEST.json").read_text())["sha256"]
        bad = [k for k in want if got.get(k) != want[k]]
        print("OK：與 MANIFEST 相同" if not bad else f"不同：{len(bad)} 個，例如 {bad[:3]}"); sys.exit(1 if bad else 0)
    stats = build(target)
    (target / "EXTRA-MANIFEST.json").write_text(json.dumps({"generator": "tools/build_story_extra.py", "seed": SEED,
        "stats": stats, "caveat": "合成資料，不含真實人物。", "sha256": manifest(target)}, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
