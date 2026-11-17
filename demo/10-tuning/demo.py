"""10｜scikit-learn GridSearchCV：一次試完一整格參數，用交叉驗證挑最好的。"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CATEGORICAL = ["event_type", "location", "source"]

df = pd.read_csv("story/data/factory/alerts.csv")
df["occurred_at"] = pd.to_datetime(df["occurred_at"])
train = df[df["occurred_at"] < "2025-12-29"]          # 前 8 週；最後 2 週的考卷不碰
X, y, groups = train[NUMERIC + CATEGORICAL], train["label"], train["incident_id"]

pre = ColumnTransformer([("num", "passthrough", NUMERIC),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

print("== 1. Pipeline 的步驟名：make_pipeline 自動取的 ==")
model = make_pipeline(pre, DecisionTreeClassifier(random_state=0))
print(list(model.named_steps))
print([k for k in model.get_params() if k.startswith("decisiontreeclassifier__max")])

print("\n== 2. GridSearchCV：5 × 2 組參數，每組考 5 折 ==")
grid = {"decisiontreeclassifier__max_depth": [2, 3, 4, 6, None],
        "decisiontreeclassifier__min_samples_leaf": [1, 20]}
search = GridSearchCV(model, grid, scoring="average_precision", cv=cv, n_jobs=1)
search.fit(X, y, groups=groups)
print(f"試了 {len(search.cv_results_['params'])} 組，共 fit {len(search.cv_results_['params']) * 5} + 1 次")

print("\n== 3. best_params_、best_score_ ==")
print("best_params_：", search.best_params_)
print(f"best_score_ ：{search.best_score_:.3f}（5 折 average precision 的平均）")

print("\n== 4. cv_results_：每一組的成績 ==")
res = pd.DataFrame(search.cv_results_)
res = res.rename(columns={"param_decisiontreeclassifier__max_depth": "max_depth",
                          "param_decisiontreeclassifier__min_samples_leaf": "min_samples_leaf"})
cols = ["max_depth", "min_samples_leaf", "mean_test_score", "std_test_score", "rank_test_score"]
print(res[cols].sort_values("rank_test_score").to_string(index=False, float_format="{:.3f}".format))

print("\n== 5. best_estimator_：用最佳參數在全部前 8 週重新 fit 好的模型 ==")
best = search.best_estimator_
print("型別：", type(best).__name__, "；樹的深度：", best[-1].get_depth(), "；葉子：", best[-1].get_n_leaves())
