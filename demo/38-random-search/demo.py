"""38｜scikit-learn RandomizedSearchCV：參數不排成格子，而是從範圍裡隨機抽；預算（n_iter）自己定。"""
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

print("== 1. 參數範圍寫成分布：randint 和 loguniform ==")
print("randint(2, 8)       ：", randint(2, 8).rvs(size=8, random_state=0).tolist())
lr_log = loguniform(0.01, 0.3).rvs(size=1000, random_state=0)
lr_uni = uniform(0.01, 0.29).rvs(size=1000, random_state=0)   # uniform(起點, 寬度)：0.01～0.30
bins = [0.01, 0.03, 0.1, 0.3]
print("抽 1000 次，落在各段的次數：")
print(pd.DataFrame({"loguniform(0.01, 0.3)": np.histogram(lr_log, bins)[0],
                    "uniform(0.01, 0.30)": np.histogram(lr_uni, bins)[0]},
                   index=["0.01–0.03", "0.03–0.1", "0.1–0.3"]).to_string())

NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CATEGORICAL = ["event_type", "location", "source"]
df = pd.read_csv("story/data/factory/alerts.csv")
df["occurred_at"] = pd.to_datetime(df["occurred_at"])
train = df[df["occurred_at"] < "2025-12-29"]          # 前 8 週，和 10 節一樣
X, y, groups = train[NUMERIC + CATEGORICAL], train["label"], train["incident_id"]
pre = ColumnTransformer([("num", "passthrough", NUMERIC),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
model = make_pipeline(pre, DecisionTreeClassifier(random_state=0))
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

print("\n== 2. 同樣試 10 組：GridSearchCV 和 RandomizedSearchCV ==")
grid = {"decisiontreeclassifier__max_depth": [2, 3, 4, 6, None],
        "decisiontreeclassifier__min_samples_leaf": [1, 20]}
gs = GridSearchCV(model, grid, scoring="average_precision", cv=cv, n_jobs=1).fit(X, y, groups=groups)
res = pd.DataFrame(gs.cv_results_)
print(f"GridSearchCV      ：max_depth 試過 {res['param_decisiontreeclassifier__max_depth'].nunique(dropna=False)} 種、"
      f"min_samples_leaf 試過 {res['param_decisiontreeclassifier__min_samples_leaf'].nunique()} 種；最佳 {gs.best_score_:.3f}")
dist = {"decisiontreeclassifier__max_depth": randint(2, 12),
        "decisiontreeclassifier__min_samples_leaf": randint(1, 80)}
rs = RandomizedSearchCV(model, dist, n_iter=10, scoring="average_precision", cv=cv,
                        random_state=0, n_jobs=1).fit(X, y, groups=groups)
res = pd.DataFrame(rs.cv_results_).rename(columns=lambda c: c.replace("param_decisiontreeclassifier__", ""))
print(f"RandomizedSearchCV：max_depth 試過 {res['max_depth'].nunique()} 種、"
      f"min_samples_leaf 試過 {res['min_samples_leaf'].nunique()} 種；最佳 {rs.best_score_:.3f}")
print("RandomizedSearchCV 前 5 名：")
cols = ["max_depth", "min_samples_leaf", "mean_test_score", "std_test_score"]
print(res.sort_values("rank_test_score")[cols].head(5).to_string(index=False, float_format="{:.3f}".format))

print("\n== 3. 預算怎麼算：每多試一組就多 fit 5 次 ==")
for name, n in [("GridSearchCV 3 個參數各 4 種", 4 ** 3), ("GridSearchCV 3 個參數各 6 種", 6 ** 3),
                ("RandomizedSearchCV n_iter=12", 12)]:
    print(f"{name:<28}{n:>4} 組 × 5 折 + 1 = {n * 5 + 1:>5} 次 fit")
