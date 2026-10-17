"""12｜讓 sklearn 幫你掃門檻：TunedThresholdClassifierCV（含 metadata routing 的坑）。"""
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import StratifiedGroupKFold, TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline

from common import FEATURES, TARGET, load_events, make_preprocessor, time_split

GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def neg_cost(y_true, y_pred):
    """成本越低越好；sklearn 的 scoring 是「越大越好」，所以加負號。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return -(10 * fn + 1 * fp)


train, _ = time_split(load_events())
X, y, groups = train[FEATURES], train[TARGET], train["incident_id"]
base = make_pipeline(make_preprocessor(scale=False),
                     HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_leaf_nodes=15,
                                                    random_state=0))
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

print("== 1. 直接傳 groups：報錯 ==")
tuned = TunedThresholdClassifierCV(base, scoring=make_scorer(neg_cost), thresholds=GRID, cv=cv,
                                   store_cv_results=True)
try:
    tuned.fit(X, y, groups=groups)
except ValueError as e:
    print("ValueError:", str(e).split(",")[0])

print("\n== 2. 打開 metadata routing 再傳 ==")
sklearn.set_config(enable_metadata_routing=True)
tuned.fit(X, y, groups=groups)
print(f"在整個 train（{len(train)} 筆）上 5 折分組交叉驗證")
print(f"最佳門檻 {tuned.best_threshold_:.2f}；每折平均成本 {-tuned.best_score_:.1f}")

print("\n== 3. 每個門檻的交叉驗證成本（節錄）==")
scores = dict(zip(np.round(tuned.cv_results_["thresholds"], 2), -tuned.cv_results_["scores"]))
for t in [0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70]:
    print(f"門檻 {t:.2f}：平均成本 {scores[t]:.1f}")

print("\n== 4. 它還是一個模型 ==")
print(f"predict 用的門檻：{tuned.best_threshold_:.2f}；predict_proba 照舊是 {type(tuned.estimator_[-1]).__name__} 的機率")
