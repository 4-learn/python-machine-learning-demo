"""產生 ML 課（2026 v2）兩條線的合成告警歷史。

    python tools/build_story_data.py           # 重建 story/data/ 並寫 MANIFEST.json
    python tools/build_story_data.py --check   # 重建到暫存目錄比對 MANIFEST

故事：告警平台已經試用 10 週。每一筆告警最後由值班的人處理成
resolved（真的要處理）或 dismissed（誤報）。這份處理紀錄就是 ML 的標籤。

兩條線欄位完全相同，程式只有一份；差別只在資料（data/factory、data/care）。
健康線只用房號與設備編號，不放姓名。全部資料為程式合成。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260112
START = "2025-11-03"
WEEKS = 10

LINES = {
    "factory": {
        "event_types": {"no_helmet": 0.55, "aisle_blocked": 0.45},
        "locations": {"aisle_01": 0.0, "aisle_02": 0.2, "dock_01": 0.5, "rest_01": -2.2},
        "location_p": [0.32, 0.30, 0.22, 0.16],
        "sources": ["camera_01", "camera_02", "camera_03", "camera_04"],
        "night_hours": set(range(22, 24)) | set(range(0, 6)),
        "per_day": 34,
        "people": 60, "prefix": "W",            # 員工代號
        "duration_scale": (28, 8),               # 真事件、誤報的持續秒數尺度
        "label_rule_min_sec": 25,
        "safe_location": "rest_01",         # 這裡的偵測不算違規（休息區可以脫帽）
        "hour_w": [1, 1, 1, 1, 1, 2, 6, 9, 10, 10, 9, 7, 5, 8, 9, 9, 8, 7, 5, 4, 3, 3, 2, 1],
    },
    "care": {
        "event_types": {"bed_exit": 0.6, "fall_detected": 0.4},
        "locations": {"room_301": 0.3, "room_302": 0.0, "room_303": 0.2, "lounge_01": -2.2},
        "location_p": [0.30, 0.28, 0.26, 0.16],
        "sources": ["camera_11", "camera_12", "wristband_03", "wristband_07"],
        "night_hours": set(range(21, 24)) | set(range(0, 7)),
        "per_day": 30,
        "people": 40, "prefix": "B",            # 床位代號（不放姓名）
        "duration_scale": (30, 8),
        "label_rule_min_sec": 25,
        "safe_location": "lounge_01",             # 交誼廳：有人陪同，坐下、彎腰不算跌倒
        "hour_w": [5, 5, 6, 6, 5, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5],
    },
}


def simulate(line: str, rng: np.random.Generator) -> pd.DataFrame:
    cfg = LINES[line]
    n_people = cfg["people"]
    people = pd.DataFrame({
        "person_id": [f"{cfg['prefix']}{i:03d}" for i in range(1, n_people + 1)],
        "days_on_site": rng.integers(10, 2000, n_people),
        "habit": rng.normal(0, 1, n_people),
    })
    people["habit"] -= 0.25 * np.log(people.days_on_site / 365)
    etypes, ep = list(cfg["event_types"]), list(cfg["event_types"].values())
    locs = list(cfg["locations"])
    hw = np.array(cfg["hour_w"], float); hw /= hw.sum()
    rows, incident = [], 0
    day0 = pd.Timestamp(START)
    for d in range(WEEKS * 7):
        day = day0 + pd.Timedelta(days=d)
        for _ in range(rng.poisson(cfg["per_day"] if day.dayofweek < 6 or line == "care" else cfg["per_day"] * 0.3)):
            incident += 1
            p = people.iloc[rng.integers(n_people)]
            etype = etypes[rng.choice(len(etypes), p=ep)]
            loc = locs[rng.choice(len(locs), p=cfg["location_p"])]
            hour = int(rng.choice(24, p=hw)); minute = int(rng.integers(60))
            night = hour in cfg["night_hours"]
            light = float(np.clip(rng.normal(110 if night else 600, 50 if night else 170), 10, 1200))
            logit = -1.5 + 0.9 * p.habit + cfg["locations"][loc] + (0.5 if night else 0.0)
            real = rng.random() < 1 / (1 + math.exp(-logit))
            ts, fs = cfg["duration_scale"]
            duration = float(rng.gamma(2.0, ts if real else fs))
            conf = rng.beta(7, 2.2) if real else rng.beta(2.4 + light / 400, 4.0)
            conf = float(np.clip(conf, 0.30, 0.995))
            # 「夠久才算」的判斷因人而異：每位值班人員心中的秒數不同
            cutoff = rng.normal(cfg["label_rule_min_sec"], 8)
            label = bool(real and loc != cfg["safe_location"] and duration >= cutoff)
            if rng.random() < 0.01:
                label = not label
            copies = 1 if rng.random() < 0.7 else int(rng.integers(2, 5))
            t0 = day + pd.Timedelta(hours=hour, minutes=minute)
            src = cfg["sources"][rng.integers(len(cfg["sources"]))]
            for c in range(copies):
                rows.append({
                    "incident_id": f"I{incident:05d}",
                    "occurred_at": t0 + pd.Timedelta(seconds=int(c * rng.integers(3, 40))),
                    "event_type": etype,
                    "location": loc,
                    "source": src if c == 0 else cfg["sources"][rng.integers(len(cfg["sources"]))],
                    "person_id": p.person_id,
                    "confidence": round(float(np.clip(conf + rng.normal(0, 0.015), 0.3, 0.995)), 3),
                    "duration_seconds": round(max(1.0, duration + rng.normal(0, 1.5)), 1),
                    "light_lux": round(light + rng.normal(0, 5)),
                    "days_on_site": int(p.days_on_site),
                    "label": int(label),
                })
    df = pd.DataFrame(rows).sort_values("occurred_at", kind="stable").reset_index(drop=True)
    df.insert(0, "alert_id", range(1, len(df) + 1))
    # 該人過去 30 天被確認為真的告警數（只看過去）
    hist, past = np.zeros(len(df), int), {}
    for i, (pid, t, y) in enumerate(zip(df.person_id, df.occurred_at, df.label)):
        lst = [x for x in past.get(pid, []) if t - x <= pd.Timedelta(days=30)]
        hist[i] = len(lst); past[pid] = lst
        if y:
            past[pid].append(t)
    df["prior_alerts_30d"] = hist
    df["hour"] = df.occurred_at.dt.hour
    # 處理紀錄：事後才知道（洩漏欄位）
    status = np.where(df.label == 1, "resolved", "dismissed")
    df["status"] = status
    lag = pd.to_timedelta(rng.integers(1, 20, len(df)), unit="min")
    df["acked_at"] = np.where(df.label == 1, (df.occurred_at + lag).dt.strftime("%Y-%m-%dT%H:%M:%S"),
                              np.where(rng.random(len(df)) < 0.4, (df.occurred_at + lag).dt.strftime("%Y-%m-%dT%H:%M:%S"), ""))
    df["resolved_at"] = np.where(df.label == 1, (df.occurred_at + lag + pd.to_timedelta(rng.integers(5, 120, len(df)), unit="min")).dt.strftime("%Y-%m-%dT%H:%M:%S"), "")
    df["occurred_at"] = df.occurred_at.dt.strftime("%Y-%m-%dT%H:%M:%S")
    cols = ["alert_id", "incident_id", "occurred_at", "event_type", "location", "source", "person_id",
            "confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour",
            "acked_at", "resolved_at", "status", "label"]
    return df[cols], people[["person_id", "days_on_site"]]


def write_new_events(df: pd.DataFrame, path: Path) -> None:
    """Workshop 1 用：最後 2 週挑 8 筆，轉成平台的事件格式（每個 incident 只取第一筆）。"""
    test = df[df.occurred_at >= "2025-12-29"].drop_duplicates("incident_id")
    picks = pd.concat([test[test.label == 1].sample(3, random_state=1),
                       test[test.label == 0].sample(5, random_state=1)]).sort_values("occurred_at")
    with path.open("w", encoding="utf-8") as f:
        for r in picks.itertuples():
            f.write(json.dumps({
                "event_type": r.event_type, "location": r.location, "source": r.source,
                "confidence": r.confidence, "occurred_at": r.occurred_at,
                "detail": {"duration_seconds": r.duration_seconds, "light_lux": r.light_lux,
                           "person_id": r.person_id},
                "_answer": r.status}, ensure_ascii=False) + "\n")


def build(out: Path) -> dict:
    stats = {}
    for k, line in enumerate(LINES):
        rng = np.random.default_rng(SEED + k)
        df, people = simulate(line, rng)
        (out / line).mkdir(parents=True, exist_ok=True)
        df.to_csv(out / line / "alerts.csv", index=False)
        people.to_csv(out / line / "people.csv", index=False)
        write_new_events(df, out / line / "new_events.jsonl")
        stats[line] = {"rows": len(df), "incidents": int(df.incident_id.nunique()),
                       "positive_rate": round(float(df.label.mean()), 4)}
    return stats


def manifest(out: Path) -> dict:
    return {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    if a.check:
        with tempfile.TemporaryDirectory() as t:
            build(Path(t)); got = manifest(Path(t))
        want = json.loads((ROOT / "story" / "data" / "MANIFEST.json").read_text())["sha256"]
        bad = [k for k in want if got.get(k) != want[k]]
        print("OK：與 MANIFEST 相同" if not bad else f"不同：{bad}"); sys.exit(1 if bad else 0)
    out = ROOT / "story" / "data"; stats = build(out)
    (out / "MANIFEST.json").write_text(json.dumps({
        "generator": "tools/build_story_data.py", "seed": SEED,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "stats": stats, "caveat": "合成資料，不代表任何真實工廠、機構或人員；健康線不含姓名。",
        "sha256": manifest(out)}, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
