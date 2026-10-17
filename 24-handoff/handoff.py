"""24｜交接包的小工具：凍結評測集、算指紋、成績表、已知錯誤、結構檢查。

    import sys; sys.path.insert(0, "24-handoff")
    from handoff import build_package, check_package
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import DATA, FEATURES, LEAKY, ROOT, TARGET, fit_valid_split, load_events, score_row, time_split

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template"        # 不會變的檔案：README.md、LABELING.md、score.py
CONFIG = ROOT / "config" / "notify.yaml"
EVAL_COLUMNS = ["event_id", "incident_id", *FEATURES, TARGET]
DATASET_VERSION = "notify-eval-2026.01"
ANNOTATION_VERSION = "chen-v1"
FN_COST, FP_COST = 10, 1
FILES = ["README.md", "LABELING.md", "manifest.json", "eval-set.csv", "eval-set.sha256", "dev-sample.csv",
         "scores.csv", "notify.yaml", "known-errors.csv", "score.py"]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(df: pd.DataFrame, path) -> None:
    """固定欄位順序、UTF-8、換行 \\n：同一份資料每次寫出來的位元組都一樣，指紋才會一樣。"""
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def eval_set() -> pd.DataFrame:
    """test 期間（2025-12-29～2026-01-11）的 977 筆，只留模型判斷時看得到的欄位。"""
    _, test = time_split(load_events())
    return test[EVAL_COLUMNS].sort_values("event_id").reset_index(drop=True)


def dev_sample(n_pos: int = 10, n_neg: int = 30, seed: int = 0) -> pd.DataFrame:
    """給 LLM 課調提示用的小樣本：驗證期（12-15～12-28），每個 incident 只取一筆，不含任何 test 事件。"""
    train, _ = time_split(load_events())
    _, valid = fit_valid_split(train)
    one = valid.drop_duplicates("incident_id")
    pos = one[one[TARGET] == 1].sample(n_pos, random_state=seed)
    neg = one[one[TARGET] == 0].sample(n_neg, random_state=seed)
    return pd.concat([pos, neg])[EVAL_COLUMNS].sort_values("event_id").reset_index(drop=True)


def cost(y, pred) -> int:
    y, pred = np.asarray(y), np.asarray(pred)
    return int(FN_COST * ((y == 1) & (pred == 0)).sum() + FP_COST * ((y == 0) & (pred == 1)).sum())


def predictions(ev: pd.DataFrame, config=CONFIG) -> pd.DataFrame:
    """三個 baseline 在評測集上的預測（0／1）與定版模型的機率。"""
    import yaml
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    model = joblib.load(ROOT / cfg["model"]["file"])
    proba = model.predict_proba(ev[FEATURES])[:, 1]
    return pd.DataFrame({"event_id": ev.event_id, "全部不通知（多數類）": 0,
                         "現行規則 conf≥0.7": (ev.confidence >= 0.7).astype(int),
                         f"定版模型 {cfg['model']['version']} @{cfg['threshold']['value']}":
                             (proba >= cfg["threshold"]["value"]).astype(int),
                         "proba": proba.round(3)})


def scores(ev: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    y = ev[TARGET]
    rows = [{**score_row(name, y, preds[name]), "cost": cost(y, preds[name])}
            for name in preds.columns if name not in ("event_id", "proba")]
    return pd.DataFrame(rows).assign(n=len(ev), dataset_version=DATASET_VERSION)


DIRECTION = {  # 13 節的錯誤分類與改善方向
    "標籤疑似錯誤：違反標註規則": "送陳主任複核；確認後修正標籤，評估時另列",
    "邊界：持續時間接近 25 秒": "改善追蹤讓持續時間更準；或請陳主任明確 20–30 秒怎麼判",
    "休息區高違規紀錄者": "加一條政策規則（01 節）：休息區一律不通知，不交給模型",
    "模型判斷錯：其他": "逐筆看影像（17–19 節），確認是否真的沒戴",
}


def known_errors(ev: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """定版模型在評測集上的錯誤，用 13 節同一套分類。"""
    model_col = [c for c in preds.columns if c.startswith("定版模型")][0]
    df = ev.assign(proba=preds.proba, pred=preds[model_col])
    df["錯誤"] = np.where((df.pred == 1) & (df[TARGET] == 0), "誤報",
                        np.where((df.pred == 0) & (df[TARGET] == 1), "漏報", ""))
    e = df[df.錯誤 != ""].copy()
    near_25 = e.duration_sec.between(20, 40)
    e["原因"] = np.select(
        [(e.錯誤 == "漏報") & ((e.zone == "休息區") | (e.duration_sec < 20)), (e.錯誤 == "漏報") & near_25,
         (e.錯誤 == "誤報") & (e.zone == "休息區"), (e.錯誤 == "誤報") & near_25],
        ["標籤疑似錯誤：違反標註規則", "邊界：持續時間接近 25 秒", "休息區高違規紀錄者", "邊界：持續時間接近 25 秒"],
        "模型判斷錯：其他")
    e["改善方向"] = e.原因.map(DIRECTION)
    return e[["event_id", "incident_id", "錯誤", "原因", "改善方向", "zone", "confidence", "duration_sec",
              "prior_violations_30d", TARGET, "proba"]].reset_index(drop=True)


def build_package(out) -> dict:
    """產出整個交接包，回傳 manifest。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ev = eval_set()
    write_csv(ev, out / "eval-set.csv")
    digest = sha256(out / "eval-set.csv")
    (out / "eval-set.sha256").write_text(f"{digest}  eval-set.csv\n", encoding="utf-8")
    write_csv(dev_sample(), out / "dev-sample.csv")
    preds = predictions(ev)
    write_csv(scores(ev, preds), out / "scores.csv")
    write_csv(known_errors(ev, preds), out / "known-errors.csv")
    shutil.copyfile(CONFIG, out / "notify.yaml")
    for name in ["README.md", "LABELING.md", "score.py"]:
        shutil.copyfile(TEMPLATE / name, out / name)
    manifest = {
        "dataset_version": DATASET_VERSION, "annotation_version": ANNOTATION_VERSION,
        "task": "這個事件要不要通知現場安全主管（needs_notify：1＝要、0＝不用）",
        "eval_set": {"file": "eval-set.csv", "sha256": digest, "rows": len(ev),
                     "incidents": int(ev.incident_id.nunique()), "positives": int(ev[TARGET].sum()),
                     "period": "2025-12-29～2026-01-11", "columns": EVAL_COLUMNS},
        "source": {"file": "data/events.csv", "sha256_12": sha256(DATA / "events.csv")[:12]},
        "cost": f"{FN_COST}×漏報 + {FP_COST}×誤報",
        "provenance": "合成資料（tools/build_data.py，seed 20260112），不代表任何真實工地或人員；"
                      "標籤模擬陳主任事後認定，約 2% 標錯。",
        "use": "整份是保留集：只用來比較，不用來調提示或門檻。調提示用 dev-sample.csv。",
        "frozen_at": "2026-02-11", "frozen_by": "ML 課 24 節",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def check_package(folder) -> pd.DataFrame:
    """結構檢查：檔案齊不齊、指紋對不對、欄位對不對、有沒有洩漏欄。內容是否合理要另外檢查。"""
    folder = Path(folder)
    rows = []

    def add(name, ok, detail=""):
        rows.append({"檢查": name, "結果": "通過" if ok else "不通過", "說明": detail})

    missing = [f for f in FILES if not (folder / f).exists()]
    add("檔案齊全", not missing, "缺 " + "、".join(missing) if missing else f"{len(FILES)} 個")
    if not (folder / "eval-set.csv").exists():
        return pd.DataFrame(rows)
    recorded = (folder / "eval-set.sha256").read_text(encoding="utf-8").split()[0] \
        if (folder / "eval-set.sha256").exists() else ""
    actual = sha256(folder / "eval-set.csv")
    add("指紋相符", recorded == actual, f"記錄 {recorded[:12]}／實際 {actual[:12]}")
    ev = pd.read_csv(folder / "eval-set.csv")
    add("欄位正確", list(ev.columns) == EVAL_COLUMNS,
        "多了 " + "、".join(sorted(set(ev.columns) - set(EVAL_COLUMNS))) if set(ev.columns) - set(EVAL_COLUMNS) else "")
    leaks = [c for c in ev.columns if c in LEAKY]
    add("沒有洩漏欄", not leaks, "、".join(leaks))
    add("event_id 不重複", ev.event_id.is_unique, f"{len(ev)} 筆")
    add("標籤只有 0／1", set(ev[TARGET].unique()) <= {0, 1})
    if (folder / "dev-sample.csv").exists():
        dev = pd.read_csv(folder / "dev-sample.csv")
        overlap = set(dev.incident_id) & set(ev.incident_id)
        add("dev 與評測集不共用 incident", not overlap, f"重疊 {len(overlap)} 個" if overlap else "")
    return pd.DataFrame(rows)
