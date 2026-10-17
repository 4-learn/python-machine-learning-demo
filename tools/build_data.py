"""產生「Python 機器學習實務」全課共用的合成資料。

用法：python tools/build_data.py            # 重建 data/，並寫 data/MANIFEST.json
      python tools/build_data.py --check    # 重建到暫存目錄，比對 MANIFEST，不覆寫

全部資料都是**合成**的：人員、工地、事件、文字都是程式產生，與真實人物無關。
同一個版本的程式、同一組套件，重建結果位元組相同（MANIFEST 記錄 SHA-256）。

產物：
  events.csv              10 週的偵測事件（表格段 02–14、16、21、23、24）
  events-new-weeks.csv    上線後 4 週的新事件，第 3 週起 C 工地換了攝影機（22）
  reports.csv             巡檢回報短文字與標籤（20）
  crops/                  頭部小圖：helmet／no_helmet（17）
  scenes/                 偵測場景＋YOLO 標註：helmet／head（18、19）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260112

SITES = ["A", "B", "C", "D"]
ZONES = ["高架作業區", "鋼筋區", "物料區", "出入口", "休息區"]
ZONE_P = [0.22, 0.24, 0.20, 0.14, 0.20]
ZONE_RISK = {"高架作業區": 0.6, "鋼筋區": 0.3, "物料區": 0.1, "出入口": -0.2, "休息區": 1.6}


# ---------------------------------------------------------------- 表格事件
def shift_of(hour: int) -> str:
    if 6 <= hour < 14:
        return "早班"
    if 14 <= hour < 22:
        return "中班"
    return "夜班"


def make_workers(rng: np.random.Generator, n: int = 80) -> pd.DataFrame:
    tenure = rng.integers(20, 3200, size=n)
    return pd.DataFrame({
        "worker_id": [f"W{i:03d}" for i in range(1, n + 1)],
        "home_site": rng.choice(SITES, size=n, p=[0.3, 0.3, 0.2, 0.2]),
        "tenure_days": tenure,
        # 個人習慣：有人就是比較常不戴。模型看不到這一欄，只能從行為間接學到。
        "habit": rng.normal(0, 1, size=n) - 0.25 * np.log(tenure / 365),
    })


def simulate_events(rng, workers, start, weeks, camera_change=None):
    """camera_change=(site, 日期)：該日起該工地攝影機換新，confidence 整體下降。"""
    rows = []
    incident = 0
    day0 = pd.Timestamp(start, tz="Asia/Taipei")
    for d in range(weeks * 7):
        day = day0 + pd.Timedelta(days=d)
        n_incidents = rng.poisson(48 if day.dayofweek < 6 else 14)
        for _ in range(n_incidents):
            incident += 1
            w = workers.iloc[rng.integers(len(workers))]
            site = w.home_site if rng.random() < 0.9 else rng.choice(SITES)
            zone = rng.choice(ZONES, p=ZONE_P)
            hour = int(rng.choice(np.arange(24), p=hour_weights()))
            minute = int(rng.integers(60))
            shift = shift_of(hour)
            night = shift == "夜班"
            light = float(np.clip(rng.normal(120 if night else 650, 60 if night else 180), 15, 1200))

            # 真的沒戴嗎？（偵測器只是「懷疑」）
            logit = -1.55 + 0.9 * w.habit + ZONE_RISK[zone] + (0.5 if night else 0.0)
            no_helmet = rng.random() < 1 / (1 + math.exp(-logit))
            duration = float(np.round(rng.gamma(2.0, 30 if no_helmet else 9), 1))

            if no_helmet:
                conf = rng.beta(7, 2.2)
            else:  # 誤報：帽子、陰影、反光；越暗越容易誤報、信心也越高
                conf = rng.beta(2.4 + (light / 400), 4.0)
            if camera_change and site == camera_change[0] and day >= pd.Timestamp(camera_change[1], tz="Asia/Taipei"):
                conf = conf * 0.78  # 新攝影機白平衡不同，信心整體偏低
            conf = float(np.clip(conf, 0.30, 0.995))

            # 標籤：主管事後認定「應該通知」
            needs = bool(no_helmet and zone != "休息區" and duration >= 25)
            if rng.random() < 0.02:  # 標註本身也會錯
                needs = not needs

            # 同一事件常被連續幾幀／兩支攝影機重複記錄（03 節的分組洩漏）
            copies = 1 if rng.random() < 0.7 else int(rng.integers(2, 5))
            base_t = day + pd.Timedelta(hours=hour, minutes=minute)
            for c in range(copies):
                rows.append({
                    "incident_id": f"I{incident:05d}",
                    "event_time": base_t + pd.Timedelta(seconds=int(c * rng.integers(3, 40))),
                    "worker_id": w.worker_id,
                    "site": site,
                    "zone": zone,
                    "shift": shift,
                    "hour": hour,
                    "confidence": round(float(np.clip(conf + rng.normal(0, 0.015), 0.3, 0.995)), 3),
                    "duration_sec": round(max(1.0, duration + rng.normal(0, 1.5)), 1),
                    "light_lux": round(light + rng.normal(0, 5), 0),
                    "people_nearby": int(rng.poisson(2.2)),
                    "tenure_days": int(w.tenure_days),
                    "needs_notify": int(needs),
                    "_true_no_helmet": int(no_helmet),
                })
    df = pd.DataFrame(rows).sort_values("event_time", kind="stable").reset_index(drop=True)
    return df


def hour_weights():
    w = np.array([2, 1, 1, 1, 1, 2, 6, 9, 10, 10, 9, 7, 5, 8, 9, 9, 8, 7, 6, 5, 4, 4, 3, 2], dtype=float)
    return w / w.sum()


def add_history_and_leaks(df: pd.DataFrame, rng, start_id: int = 1) -> pd.DataFrame:
    """prior_violations_30d：該員過去 30 天被認定需通知的「事件數」（只看過去，不含本次）。
    ticket_id／notified_at／handled_minutes：只有被通知過的事件才有——事後才知道的欄位。"""
    df = df.copy()
    df.insert(0, "event_id", [f"E{i:05d}" for i in range(start_id, start_id + len(df))])
    hist = np.zeros(len(df), dtype=int)
    past: dict[str, list[pd.Timestamp]] = {}
    for i, (w, t) in enumerate(zip(df.worker_id, df.event_time)):
        lst = [x for x in past.get(w, []) if t - x <= pd.Timedelta(days=30)]
        hist[i] = len(lst)
        past[w] = lst
        if df.needs_notify.iat[i]:
            past[w].append(t)
    df["prior_violations_30d"] = hist
    notified = df.needs_notify.astype(bool) & (rng.random(len(df)) < 0.97)
    df["ticket_id"] = np.where(notified, [f"T{i:05d}" for i in range(len(df))], "")
    lag = pd.to_timedelta(rng.integers(2, 15, len(df)), unit="min")
    df["notified_at"] = np.where(notified, (df.event_time + lag).dt.strftime("%Y-%m-%d %H:%M:%S%z"), "")
    df["handled_minutes"] = np.where(notified, rng.integers(5, 240, len(df)), np.nan)
    return df


COLUMNS = ["event_id", "incident_id", "event_time", "worker_id", "site", "zone", "shift", "hour",
           "confidence", "duration_sec", "light_lux", "people_nearby", "tenure_days",
           "prior_violations_30d", "ticket_id", "notified_at", "handled_minutes", "needs_notify"]


def write_events(out: Path, rng) -> dict:
    workers = make_workers(rng)
    df = add_history_and_leaks(simulate_events(rng, workers, "2025-11-03", 10), rng)
    df["event_time"] = df.event_time.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df[COLUMNS].to_csv(out / "events.csv", index=False)

    new = add_history_and_leaks(
        simulate_events(rng, workers, "2026-01-12", 4, camera_change=("C", "2026-01-26")), rng, start_id=len(df) + 1)
    new["event_time"] = new.event_time.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    # 上線後只有一部分事件被主管複查過：其餘沒有標籤
    reviewed = rng.random(len(new)) < 0.25
    new["needs_notify"] = np.where(reviewed, new.needs_notify, np.nan)
    new = new.drop(columns=["ticket_id", "notified_at", "handled_minutes"])
    new.drop(columns=["_true_no_helmet"]).to_csv(out / "events-new-weeks.csv", index=False)
    return {"events": len(df), "events_positive_rate": round(float(df.needs_notify.mean()), 4), "new_weeks": len(new)}


# ---------------------------------------------------------------- 巡檢文字
def write_reports(out: Path, rng) -> dict:
    places = ["三樓鋼筋區", "B 棟高架平台", "物料堆置區", "北側出入口", "五樓外牆鷹架", "地下室模板區", "吊掛作業區", "休息區"]
    pos = [  # 應通知
        "{p}有一名工人沒戴安全帽在作業", "{p}發現人員未配戴安全帽攀爬鷹架", "{p}工人把安全帽掛在腰上就上去施工",
        "{p}有人戴棒球帽代替安全帽", "{p}兩名外包人員頭上沒有任何防護", "{p}吊掛作業時旁邊的人沒有戴帽子",
        "{p}看到新進人員的帽子沒戴在頭上而是拿在手上", "{p}有人安全帽戴著但下巴帶沒扣，彎腰時掉了還繼續做",
        "{p}人員裸頭進入施工範圍，提醒後仍未改善", "{p}夜班有工人沒戴頭盔在搬鋼筋",
    ]
    neg = [  # 不需通知
        "{p}人員皆已戴好安全帽", "{p}巡查完畢，安全帽配戴正常", "{p}工人在休息時把安全帽放在旁邊吃便當",
        "{p}有一頂安全帽遺落在地上，已收回", "{p}安全帽庫存不足，需要補貨", "{p}新到一批安全帽，已發放",
        "{p}提醒大家安全帽要扣好下巴帶，目前都有扣", "{p}未發現未戴安全帽的情形", "{p}照明不足，建議加裝燈具",
        "{p}地面濕滑，已放置警示牌", "{p}攝影機鏡頭被灰塵遮住，請清潔",
    ]
    rows = []
    for i in range(480):
        label = int(rng.random() < 0.35)
        tpl = rng.choice(pos if label else neg)
        text = tpl.format(p=rng.choice(places))
        if rng.random() < 0.3:
            text += rng.choice(["。", "，請處理。", "，已拍照。", "（下午巡檢）", "，麻煩主管看一下。"])
        rows.append({"report_id": f"R{i + 1:04d}", "text": text, "needs_notify": label})
    df = pd.DataFrame(rows)
    df.to_csv(out / "reports.csv", index=False)
    return {"reports": len(df), "reports_positive_rate": round(float(df.needs_notify.mean()), 4)}


# ---------------------------------------------------------------- 影像
SKIN = [(241, 194, 160), (224, 172, 132), (198, 143, 103), (160, 110, 75)]
HAIR = [(25, 20, 18), (55, 40, 30), (90, 70, 50), (120, 120, 120)]
HELMET = [(245, 245, 240), (250, 205, 30), (245, 120, 20), (40, 90, 200), (210, 40, 40)]
CAPS = [(30, 30, 50), (40, 60, 40), (150, 30, 30), (200, 200, 200)]
CAMERAS = {  # 每支攝影機的背景與色偏不同——17 節依攝影機分組切割
    "cam1": ((150, 150, 145), (1.00, 1.00, 1.00)),
    "cam2": ((120, 110, 95), (1.08, 1.00, 0.90)),
    "cam3": ((95, 105, 120), (0.90, 0.98, 1.10)),
    "cam4": ((170, 160, 130), (1.00, 1.02, 0.92)),
    "cam5": ((70, 75, 80), (0.85, 0.85, 0.85)),   # 夜間
    "cam6": ((135, 145, 120), (0.95, 1.05, 0.95)),
}


def draw_head(d: ImageDraw.ImageDraw, cx, cy, r, kind, rng):
    """kind: helmet / bare / cap。回傳頭部（含帽子）的外框。"""
    skin = SKIN[rng.integers(len(SKIN))]
    d.ellipse([cx - r, cy - r, cx + r, cy + r * 1.15], fill=skin)
    top = cy - r
    if kind == "helmet":
        col = HELMET[rng.integers(len(HELMET))]
        d.chord([cx - r * 1.15, cy - r * 1.35, cx + r * 1.15, cy + r * 0.55], 180, 360, fill=col)
        d.rectangle([cx - r * 1.3, cy - r * 0.42, cx + r * 1.3, cy - r * 0.25], fill=col)
        d.line([cx, cy - r * 1.3, cx, cy - r * 0.45], fill=tuple(int(c * 0.8) for c in col), width=max(1, int(r / 6)))
        top = cy - r * 1.35
    elif kind == "cap":
        col = CAPS[rng.integers(len(CAPS))]
        d.chord([cx - r * 1.02, cy - r * 1.15, cx + r * 1.02, cy + r * 0.3], 180, 360, fill=col)
        side = 1 if rng.random() < 0.5 else -1
        d.polygon([(cx, cy - r * 0.45), (cx + side * r * 1.6, cy - r * 0.3), (cx + side * r * 1.5, cy - r * 0.15), (cx, cy - r * 0.25)], fill=col)
        top = cy - r * 1.15
    else:
        hair = HAIR[rng.integers(len(HAIR))]
        d.chord([cx - r * 1.03, cy - r * 1.08, cx + r * 1.03, cy + r * 0.2], 180, 360, fill=hair)
        top = cy - r * 1.08
    return [cx - r * 1.3, top, cx + r * 1.3, cy + r * 1.15]


def camera_finish(img: Image.Image, cast, rng, night=False):
    a = np.asarray(img, dtype=np.float32) * np.array(cast, dtype=np.float32)
    a = a * rng.uniform(0.75, 1.15) * (0.55 if night else 1.0)
    a = a + rng.normal(0, 9 if night else 4, a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4)))
    return img


def write_crops(out: Path, rng) -> dict:
    """96×96 頭部小圖。no_helmet 包含裸頭與棒球帽（帽子不是安全帽）。"""
    root = out / "crops"
    rows = []
    for cam, (bg, cast) in CAMERAS.items():
        for i in range(110):
            label = "helmet" if rng.random() < 0.5 else "no_helmet"
            kind = "helmet" if label == "helmet" else ("cap" if rng.random() < 0.3 else "bare")
            img = Image.new("RGB", (96, 96), tuple(int(c + rng.normal(0, 12)) for c in bg))
            d = ImageDraw.Draw(img)
            for _ in range(3):  # 背景雜物
                x, y = rng.integers(0, 96, 2)
                d.rectangle([x, y, x + rng.integers(5, 30), y + rng.integers(5, 30)], fill=tuple(int(v) for v in rng.integers(40, 200, 3)))
            d.rectangle([20, 70, 76, 96], fill=tuple(int(v) for v in rng.integers(30, 160, 3)))  # 肩膀
            draw_head(d, 48 + rng.integers(-6, 7), 50 + rng.integers(-5, 6), int(rng.integers(17, 23)), kind, rng)
            img = img.rotate(float(rng.normal(0, 8)), fillcolor=bg)
            img = camera_finish(img, cast, rng, night=(cam == "cam5"))
            name = f"{cam}_{i:03d}.png"
            (root / label).mkdir(parents=True, exist_ok=True)
            img.save(root / label / name.replace(".png", ".jpg"), quality=90)
            rows.append({"file": f"{label}/{name.replace('.png', '.jpg')}", "camera": cam, "label": label, "kind": kind})
    pd.DataFrame(rows).to_csv(root / "index.csv", index=False)
    return {"crops": len(rows)}


def write_scenes(out: Path, rng) -> dict:
    """320×320 場景，YOLO 格式標註：class 0 = helmet（戴著安全帽的頭），1 = head（沒戴的頭）。"""
    root = out / "scenes"
    counts = {}
    for split, n in [("train", 360), ("val", 90), ("test", 90)]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cam = list(CAMERAS)[rng.integers(len(CAMERAS))]
            bg, cast = CAMERAS[cam]
            img = Image.new("RGB", (320, 320), bg)
            d = ImageDraw.Draw(img)
            d.rectangle([0, 230, 320, 320], fill=tuple(int(c * 0.8) for c in bg))  # 地面
            for _ in range(6):
                x, y = rng.integers(0, 320, 2)
                d.rectangle([x, y, x + rng.integers(10, 60), y + rng.integers(5, 40)], fill=tuple(int(v) for v in rng.integers(40, 200, 3)))
            labels = []
            xs = rng.permutation(np.arange(40, 300, 62))[: rng.integers(1, 5)]
            for x in xs:
                r = int(rng.integers(10, 17))
                cy = int(rng.integers(80, 170))
                kind = "helmet" if rng.random() < 0.6 else ("cap" if rng.random() < 0.25 else "bare")
                body = tuple(int(v) for v in rng.integers(30, 200, 3))
                d.rectangle([x - r * 1.5, cy + r * 1.1, x + r * 1.5, cy + r * 6], fill=body)
                d.rectangle([x - r * 1.2, cy + r * 6, x - r * 0.2, cy + r * 9], fill=(50, 50, 70))
                d.rectangle([x + r * 0.2, cy + r * 6, x + r * 1.2, cy + r * 9], fill=(50, 50, 70))
                box = draw_head(d, int(x), cy, r, kind, rng)
                cls = 0 if kind == "helmet" else 1
                x0, y0, x1, y1 = [max(0, min(320, v)) for v in box]
                labels.append(f"{cls} {(x0 + x1) / 640:.6f} {(y0 + y1) / 640:.6f} {(x1 - x0) / 320:.6f} {(y1 - y0) / 320:.6f}")
            if rng.random() < 0.25:  # 干擾物：地上的安全帽不是「戴著」——不標註
                hx, hy = int(rng.integers(30, 290)), int(rng.integers(250, 300))
                col = HELMET[rng.integers(len(HELMET))]
                d.chord([hx - 14, hy - 14, hx + 14, hy + 10], 180, 360, fill=col)
            img = camera_finish(img, cast, rng, night=(cam == "cam5"))
            img.save(root / "images" / split / f"{split}_{i:03d}.jpg", quality=90)
            (root / "labels" / split / f"{split}_{i:03d}.txt").write_text("\n".join(labels) + "\n")
        counts[split] = n
    (root / "helmet.yaml").write_text(
        "# 由 tools/build_data.py 產生。path 請用絕對路徑或在 data/scenes 目錄下執行。\n"
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: helmet\n  1: head\n")
    return counts


# ---------------------------------------------------------------- 主程式
def build(out: Path) -> dict:
    rng = np.random.default_rng(SEED)
    out.mkdir(parents=True, exist_ok=True)
    for p in ["crops", "scenes"]:
        shutil.rmtree(out / p, ignore_errors=True)
    stats = {}
    stats.update(write_events(out, rng))
    stats.update(write_reports(out, np.random.default_rng(SEED + 1)))
    stats.update(write_crops(out, np.random.default_rng(SEED + 2)))
    stats["scenes"] = write_scenes(out, np.random.default_rng(SEED + 3))
    return stats


def manifest(out: Path) -> dict:
    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json" and not p.name.startswith("photo-") and not p.name.endswith(".cache") and p.suffix != ".pt":
            files[str(p.relative_to(out))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    import PIL
    env = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "pillow": PIL.__version__}
    if a.check:
        with tempfile.TemporaryDirectory() as t:
            build(Path(t))
            got = manifest(Path(t))
        want = json.loads((ROOT / "data" / "MANIFEST.json").read_text())["sha256"]
        bad = [k for k in want if got.get(k) != want[k]]
        print("OK：與 MANIFEST 相同" if not bad else f"不同：{len(bad)} 個檔案，例如 {bad[:3]}")
        sys.exit(1 if bad else 0)
    out = ROOT / "data"
    stats = build(out)
    (out / "MANIFEST.json").write_text(json.dumps(
        {"generator": "tools/build_data.py", "seed": SEED, "environment": env, "stats": stats,
         "caveat": "合成資料，不代表任何真實工地或人員。", "sha256": manifest(out)},
        ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
