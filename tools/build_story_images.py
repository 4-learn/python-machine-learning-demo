"""產生 ML 課（2026 v2）16–24 節兩條線的影像資料。

    python tools/build_story_images.py            # 重建 story/data/{factory,care}/{crops,scenes}
    python tools/build_story_images.py --check    # 重建到暫存目錄比對 MANIFEST

兩條線同一種格式，程式只有一份：
  crops/{ok,alert}/camX_NNN.jpg ＋ crops/index.csv   96×96 小圖（分類用，16–19 節）
  scenes/images/{train,val,test}/*.jpg ＋ labels/     320×320 場景，YOLO 格式（20–23 節）
  scenes/meta.csv                                    test 場景的拍攝資訊（送事件用，22–23 節）
  scenes/data.yaml                                   ultralytics 的資料設定
類別：0＝沒事（helmet／person），1＝要注意（head／fallen）。

工廠線：戴著的安全帽是 helmet；裸頭和棒球帽是 head（帽子不是安全帽）；地上的安全帽不標。
健康線：站著、坐著、躺在床上是 person；倒在地上是 fallen（床上的人不算跌倒）。
全部是程式畫出來的合成圖，不是真實人物。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260118

SKIN = [(241, 194, 160), (224, 172, 132), (198, 143, 103), (160, 110, 75)]
HAIR = [(25, 20, 18), (55, 40, 30), (90, 70, 50), (190, 190, 190)]
HELMET = [(245, 245, 240), (250, 205, 30), (245, 120, 20), (40, 90, 200)]
CAPS = [(30, 30, 50), (40, 60, 40), (150, 30, 30)]
CLOTHES = [(70, 110, 160), (160, 90, 90), (90, 140, 100), (180, 170, 150), (120, 100, 150)]
CAMERAS = {
    "factory": {"camera_01": ((150, 150, 145), 1.0), "camera_02": ((120, 110, 95), 1.0),
                "camera_03": ((95, 105, 120), 1.0), "camera_04": ((70, 75, 80), 0.55)},   # camera_04 夜間
    "care": {"camera_11": ((205, 195, 175), 1.0), "camera_12": ((190, 200, 205), 1.0),
             "camera_13": ((200, 185, 160), 1.0), "camera_14": ((120, 110, 95), 0.5)},   # camera_14 夜間
}
LOCATIONS = {"factory": ["aisle_01", "aisle_02", "dock_01", "rest_01"],
             "care": ["room_301", "room_302", "room_303", "lounge_01"]}
SAFE = {"factory": "rest_01", "care": "lounge_01"}
NAMES = {"factory": ["helmet", "head"], "care": ["person", "fallen"]}
EVENT = {"factory": "no_helmet", "care": "fall_detected"}


def c(rng, base, jitter=12):
    return tuple(int(np.clip(v + rng.normal(0, jitter), 0, 255)) for v in base)


def finish(img, gain, rng):
    a = np.asarray(img, dtype=np.float32) * gain * rng.uniform(0.85, 1.1)
    a = a + rng.normal(0, 4 if gain >= 1 else 9, a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.2)))
    return img


# ---------------------------------------------------------------- 工廠線：頭
def draw_head(d, cx, cy, r, kind, rng):
    d.ellipse([cx - r, cy - r, cx + r, cy + r * 1.15], fill=SKIN[rng.integers(len(SKIN))])
    if kind == "helmet":
        col = HELMET[rng.integers(len(HELMET))]
        d.chord([cx - r * 1.15, cy - r * 1.35, cx + r * 1.15, cy + r * 0.55], 180, 360, fill=col)
        d.rectangle([cx - r * 1.3, cy - r * 0.42, cx + r * 1.3, cy - r * 0.25], fill=col)
        top = cy - r * 1.35
    elif kind == "cap":
        col = CAPS[rng.integers(len(CAPS))]
        d.chord([cx - r * 1.02, cy - r * 1.15, cx + r * 1.02, cy + r * 0.3], 180, 360, fill=col)
        s = 1 if rng.random() < 0.5 else -1
        d.polygon([(cx, cy - r * .45), (cx + s * r * 1.6, cy - r * .3), (cx + s * r * 1.5, cy - r * .15), (cx, cy - r * .25)], fill=col)
        top = cy - r * 1.15
    else:
        d.chord([cx - r * 1.03, cy - r * 1.08, cx + r * 1.03, cy + r * 0.2], 180, 360, fill=HAIR[rng.integers(3)])
        top = cy - r * 1.08
    return [cx - r * 1.3, top, cx + r * 1.3, cy + r * 1.15]


# ---------------------------------------------------------------- 健康線：人
def draw_person(d, x, y, s, pose, rng, bed=None):
    """pose：stand／sit／bed／fallen。回傳整個人的外框。(x, y) 是腳底中心（fallen 是身體中心）。"""
    skin, cloth, hair = SKIN[rng.integers(len(SKIN))], CLOTHES[rng.integers(len(CLOTHES))], HAIR[rng.integers(len(HAIR))]
    pants = (60, 60, 70)
    if pose == "stand":
        d.rectangle([x - s * .35, y - s * 1.9, x + s * .35, y - s * 1.0], fill=cloth)       # 身體
        d.rectangle([x - s * .3, y - s * 1.0, x - s * .05, y], fill=pants)
        d.rectangle([x + s * .05, y - s * 1.0, x + s * .3, y], fill=pants)
        d.ellipse([x - s * .25, y - s * 2.4, x + s * .25, y - s * 1.9], fill=skin)
        d.chord([x - s * .26, y - s * 2.42, x + s * .26, y - s * 2.1], 180, 360, fill=hair)
        return [x - s * .4, y - s * 2.42, x + s * .4, y]
    if pose == "sit":
        d.rectangle([x - s * .55, y - s * .9, x + s * .25, y - s * .8], fill=(140, 110, 80))  # 椅子
        d.rectangle([x - s * .55, y - s * .8, x - s * .48, y], fill=(140, 110, 80))
        d.rectangle([x - s * .45, y - s * 1.75, x + s * .1, y - s * .9], fill=cloth)
        d.rectangle([x - s * .1, y - s * 1.0, x + s * .55, y - s * .8], fill=pants)          # 大腿
        d.rectangle([x + s * .4, y - s * .9, x + s * .55, y], fill=pants)                     # 小腿
        d.ellipse([x - s * .42, y - s * 2.25, x + s * .08, y - s * 1.75], fill=skin)
        d.chord([x - s * .43, y - s * 2.27, x + s * .09, y - s * 1.95], 180, 360, fill=hair)
        return [x - s * .58, y - s * 2.27, x + s * .58, y]
    # bed／fallen：身體橫躺
    if pose == "bed":
        bx0, by0, bx1, by1 = bed
        d.rectangle(bed, fill=(235, 235, 240))
        d.rectangle([bx0, by1, bx1, by1 + s * .25], fill=(130, 100, 70))
        x, y = (bx0 + bx1) / 2, (by0 + by1) / 2
    w = s * 2.2
    d.rectangle([x - w / 2 + s * .5, y - s * .3, x + w * .15, y + s * .3], fill=cloth)
    d.rectangle([x + w * .15, y - s * .25, x + w / 2, y + s * .25], fill=pants)
    d.ellipse([x - w / 2, y - s * .26, x - w / 2 + s * .52, y + s * .26], fill=skin)
    d.chord([x - w / 2 - s * .02, y - s * .28, x - w / 2 + s * .25, y + s * .28], 90, 270, fill=hair)
    if pose == "bed":
        d.rectangle([x - w * .1, y - s * .45, x + w / 2 + 4, y + s * .45], fill=(180, 190, 220))  # 被子
        return [x - w / 2, y - s * .45, x + w / 2 + 4, y + s * .45]
    return [x - w / 2, y - s * .3, x + w / 2, y + s * .3]


def scene_background(line, cam, rng, size):
    bg, _ = CAMERAS[line][cam]
    img = Image.new("RGB", (size, size), c(rng, bg, 6))
    d = ImageDraw.Draw(img)
    floor = int(size * .55)
    d.rectangle([0, floor, size, size], fill=tuple(int(v * .82) for v in bg))
    for _ in range(4 if size > 100 else 2):
        x, y = rng.integers(0, size, 2)
        d.rectangle([x, y, x + rng.integers(6, size // 5), y + rng.integers(4, size // 8)],
                    fill=tuple(int(v) for v in rng.integers(60, 200, 3)))
    return img, d, floor


# ---------------------------------------------------------------- 小圖
def write_crops(line, out, rng):
    rows = []
    for cam in CAMERAS[line]:
        for i in range(100):
            alert = rng.random() < 0.5
            img, d, floor = scene_background(line, cam, rng, 96)
            if line == "factory":
                kind = "helmet" if not alert else ("cap" if rng.random() < .3 else "bare")
                d.rectangle([20, 70, 76, 96], fill=CLOTHES[rng.integers(len(CLOTHES))])
                draw_head(d, 48 + rng.integers(-6, 7), 50 + rng.integers(-5, 6), int(rng.integers(17, 23)), kind, rng)
            else:
                kind = "fallen" if alert else ["stand", "sit", "bed"][rng.choice(3, p=[.4, .3, .3])]
                if kind == "fallen":
                    draw_person(d, 48 + rng.integers(-5, 6), 70 + rng.integers(-6, 6), 18, "fallen", rng)
                elif kind == "bed":
                    draw_person(d, 0, 0, 18, "bed", rng, bed=[6, 50 + int(rng.integers(-6, 6)), 90, 80])
                else:
                    draw_person(d, 48 + rng.integers(-8, 9), 90, 20, kind, rng)
            img = img.rotate(float(rng.normal(0, 6)), fillcolor=CAMERAS[line][cam][0])
            img = finish(img, CAMERAS[line][cam][1], rng)
            label = "alert" if alert else "ok"
            name = f"{cam}_{i:03d}.jpg"
            (out / "crops" / label).mkdir(parents=True, exist_ok=True)
            img.save(out / "crops" / label / name, quality=90)
            rows.append({"file": f"{label}/{name}", "camera": cam, "label": label, "kind": kind})
    pd.DataFrame(rows).to_csv(out / "crops" / "index.csv", index=False)
    return len(rows)


# ---------------------------------------------------------------- 場景
def write_scenes(line, out, rng):
    root = out / "scenes"
    meta = []
    counts = {}
    for split, n in [("train", 240), ("val", 60), ("test", 60)]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cam = list(CAMERAS[line])[rng.integers(4)]
            img, d, floor = scene_background(line, cam, rng, 320)
            labels, any_alert = [], False
            slots = rng.permutation(np.arange(50, 300, 70))[: rng.integers(1, 4)]
            for x in slots:
                if line == "factory":
                    r = int(rng.integers(10, 16)); cy = int(rng.integers(80, 160))
                    kind = "helmet" if rng.random() < .6 else ("cap" if rng.random() < .25 else "bare")
                    d.rectangle([x - r * 1.5, cy + r * 1.1, x + r * 1.5, cy + r * 6], fill=CLOTHES[rng.integers(len(CLOTHES))])
                    d.rectangle([x - r * 1.2, cy + r * 6, x + r * 1.2, cy + r * 9], fill=(50, 50, 70))
                    box = draw_head(d, int(x), cy, r, kind, rng); cls = 0 if kind == "helmet" else 1
                else:
                    s = int(rng.integers(26, 36))
                    kind = ["stand", "sit", "bed", "fallen"][rng.choice(4, p=[.35, .2, .15, .3])]
                    if kind == "fallen":
                        box = draw_person(d, int(x), int(rng.integers(floor + 30, 300)), s, "fallen", rng)
                    elif kind == "bed":
                        by = int(rng.integers(floor - 30, floor + 10))
                        box = draw_person(d, 0, 0, s, "bed", rng, bed=[int(x) - 45, by, int(x) + 45, by + 40])
                    else:
                        box = draw_person(d, int(x), int(rng.integers(floor + 40, 310)), s, kind, rng)
                    cls = 1 if kind == "fallen" else 0
                any_alert |= cls == 1
                x0, y0, x1, y1 = [max(0, min(320, v)) for v in box]
                labels.append(f"{cls} {(x0 + x1) / 640:.6f} {(y0 + y1) / 640:.6f} {(x1 - x0) / 320:.6f} {(y1 - y0) / 320:.6f}")
            if rng.random() < .25:   # 干擾物：地上的安全帽（工廠）、空的輪椅（健康），不標
                hx, hy = int(rng.integers(30, 290)), int(rng.integers(floor + 20, 300))
                if line == "factory":
                    d.chord([hx - 14, hy - 14, hx + 14, hy + 10], 180, 360, fill=HELMET[rng.integers(len(HELMET))])
                else:
                    d.ellipse([hx - 14, hy - 14, hx + 14, hy + 14], outline=(60, 60, 60), width=3)
                    d.rectangle([hx - 10, hy - 30, hx + 10, hy - 14], fill=(120, 40, 40))
            img = finish(img, CAMERAS[line][cam][1], rng)
            stem = f"{split}_{i:03d}"
            img.save(root / "images" / split / f"{stem}.jpg", quality=90)
            (root / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + "\n")
            if split == "test":
                loc = LOCATIONS[line][rng.choice(4, p=[.3, .3, .25, .15])]
                night = CAMERAS[line][cam][1] < 1
                duration = float(np.round(rng.gamma(2.0, 30 if any_alert else 9), 1))
                meta.append({"image": f"{stem}.jpg", "source": cam, "location": loc,
                             "occurred_at": str(pd.Timestamp("2026-01-05") + pd.Timedelta(minutes=int(i * 97 % 10080))),
                             "duration_seconds": duration, "light_lux": int(rng.normal(110 if night else 600, 40)),
                             "person_id": f"{'W' if line == 'factory' else 'B'}{int(rng.integers(1, 41)):03d}",
                             "should_alert": int(any_alert and loc != SAFE[line] and duration >= 25)})
        counts[split] = n
    pd.DataFrame(meta).to_csv(root / "meta.csv", index=False)
    (root / "data.yaml").write_text(
        f"# 由 tools/build_story_images.py 產生；在 scenes 資料夾裡用相對路徑\npath: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: {NAMES[line][0]}\n  1: {NAMES[line][1]}\n")
    return counts


def build(base: Path) -> dict:
    stats = {}
    for k, line in enumerate(["factory", "care"]):
        out = base / line
        for sub in ["crops", "scenes"]:
            shutil.rmtree(out / sub, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        stats[line] = {"crops": write_crops(line, out, np.random.default_rng(SEED + 10 * k)),
                       "scenes": write_scenes(line, out, np.random.default_rng(SEED + 10 * k + 1))}
    return stats


def manifest(base: Path) -> dict:
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.rglob("*")) if p.is_file() and ("/crops/" in str(p) or "/scenes/" in str(p))
            and not p.name.endswith(".cache")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    target = ROOT / "story" / "data"
    if a.check:
        with tempfile.TemporaryDirectory() as t:
            build(Path(t)); got = manifest(Path(t))
        want = json.loads((target / "IMAGES-MANIFEST.json").read_text())["sha256"]
        bad = [k for k in want if got.get(k) != want[k]]
        print("OK：與 MANIFEST 相同" if not bad else f"不同：{len(bad)} 個，例如 {bad[:3]}"); sys.exit(1 if bad else 0)
    stats = build(target)
    import PIL
    (target / "IMAGES-MANIFEST.json").write_text(json.dumps({
        "generator": "tools/build_story_images.py", "seed": SEED,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__},
        "stats": stats, "caveat": "程式畫出的合成圖，不是真實人物。", "sha256": manifest(target)},
        ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
