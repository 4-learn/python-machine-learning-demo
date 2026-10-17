"""產生 ML 段的「定版」產物，給 21–24 節直接使用（學員在 10、12、14 節會自己做出同樣的東西）。

    python tools/build_artifacts.py

產物：
  models/notify-hgb.joblib   10 節調參後的 HistGradientBoosting（前 8 週訓練）
  config/notify.yaml         12 節選出的門檻與依據（漏報代價 = 10 × 誤報）
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, fit_valid_split, load_events, make_preprocessor, time_split, xy  # noqa: E402

PARAMS = {"learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 15}  # 10 節 GridSearchCV 的結果
FN_COST, FP_COST = 10, 1


def make_model():
    return make_pipeline(make_preprocessor(scale=False), HistGradientBoostingClassifier(random_state=0, **PARAMS))


def choose_threshold(model, valid):
    X, y = xy(valid)
    proba = model.predict_proba(X)[:, 1]
    best = None
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        tn, fp, fn, tp = confusion_matrix(y, (proba >= t).astype(int), labels=[0, 1]).ravel()
        cost = FP_COST * fp + FN_COST * fn
        if best is None or cost < best[1]:
            best = (float(t), int(cost))
    return best


def main():
    events = load_events()
    train, _ = time_split(events)
    fit, valid = fit_valid_split(train)

    # 門檻：在「前 6 週」訓練的模型上，用驗證集（第 7–8 週）挑
    probe = make_model().fit(*xy(fit))
    threshold, cost = choose_threshold(probe, valid)

    # 定版模型：用全部 8 週重新訓練，門檻沿用
    model = make_model().fit(*xy(train))
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "config").mkdir(exist_ok=True)
    path = ROOT / "models" / "notify-hgb.joblib"
    joblib.dump(model, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    data_digest = hashlib.sha256((ROOT / "data" / "events.csv").read_bytes()).hexdigest()[:12]

    (ROOT / "config" / "notify.yaml").write_text(
        "# 由 tools/build_artifacts.py 產生；改門檻要經過 12 節的流程並由陳主任核准。\n"
        "model:\n"
        f"  file: models/notify-hgb.joblib\n"
        f"  sha256_12: {digest}\n"
        f"  version: notify-hgb-2026.01\n"
        f"  sklearn: {sklearn.__version__}\n"
        f"  trained_on: data/events.csv（sha256_12 {data_digest}）2025-11-03～2025-12-28\n"
        "threshold:\n"
        f"  value: {threshold}\n"
        f"  chosen_on: 驗證集 2025-12-15～2025-12-28（{len(valid)} 筆）\n"
        f"  rule: 最小化 {FN_COST}×漏報 + {FP_COST}×誤報（驗證集成本 {cost}）\n"
        "  approved_by: 陳主任\n"
        "  approved_at: 2026-01-13\n"
        "camera:\n"
        "  detector: models/helmet-yolo11n.pt\n"
        "  min_confidence: 0.25\n",
        encoding="utf-8")
    print(f"threshold={threshold} cost={cost} model_sha256_12={digest}")


if __name__ == "__main__":
    main()
