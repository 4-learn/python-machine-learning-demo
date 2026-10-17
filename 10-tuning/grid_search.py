"""10｜GridSearchCV：在 train 上用交叉驗證挑梯度提升的參數。這支程式完全不讀 test。"""
import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from common import load_events, make_preprocessor, time_split, xy

threadpool_limits(1)  # 4GB 學員機：單執行緒、n_jobs=1，記憶體約 200MB

train, _ = time_split(load_events())
X, y = xy(train)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
pipe = make_pipeline(make_preprocessor(scale=False), HistGradientBoostingClassifier(random_state=0))
grid = {
    "histgradientboostingclassifier__learning_rate": [0.05, 0.1, 0.2],
    "histgradientboostingclassifier__max_depth": [3, 5, None],
    "histgradientboostingclassifier__max_leaf_nodes": [15, 31],
}
n = 1
for values in grid.values():
    n *= len(values)
print("== 1. 網格大小 ==")
print(f"{n} 組參數 × 5 折 = {n * 5} 次訓練")

start = time.perf_counter()
search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=1)
search.fit(X, y, groups=train["incident_id"])
seconds = time.perf_counter() - start

res = pd.DataFrame(search.cv_results_)
res = res.rename(columns=lambda c: c.replace("param_histgradientboostingclassifier__", ""))
res["max_depth"] = [str(p["histgradientboostingclassifier__max_depth"]) for p in search.cv_results_["params"]]
cols = ["rank_test_score", "learning_rate", "max_depth", "max_leaf_nodes", "mean_test_score", "std_test_score"]
print("\n== 2. 前 6 名（依 CV 的 average precision）==")
print(res.sort_values(["rank_test_score", "max_leaf_nodes"])[cols].head(6).to_string(index=False, float_format="%.3f"))

print("\n== 3. 最佳參數 ==")
best = {k.split("__")[1]: v for k, v in search.best_params_.items()}
print(best)
print(f"CV average precision {search.best_score_:.3f}")
default = res[(res.learning_rate == 0.1) & (res.max_depth == "None") & (res.max_leaf_nodes == 31)]
print(f"sklearn 預設（0.1, None, 31）的 CV average precision {default.mean_test_score.iloc[0]:.3f}")
print(f"\n（本機耗時 {seconds:.0f} 秒）")
