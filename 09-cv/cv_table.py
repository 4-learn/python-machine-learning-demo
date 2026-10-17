"""09｜交叉驗證：在整個 train（前 8 週）上做 5 折，成績表改成「平均 ± 標準差」。"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from common import load_events, make_preprocessor, time_split, xy

threadpool_limits(2)

train, _ = time_split(load_events())
X, y = xy(train)
groups = train["incident_id"]
SCORING = ["accuracy", "precision", "recall"]
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)


def hgb():
    return make_pipeline(make_preprocessor(scale=False), HistGradientBoostingClassifier(random_state=0))


print("== 1. 梯度提升：5 折各自的分數 ==")
res = cross_validate(hgb(), X, y, groups=groups, cv=cv, scoring=SCORING)
folds = pd.DataFrame({m: res[f"test_{m}"] for m in SCORING})
folds.index = [f"第 {i + 1} 折" for i in range(5)]
folds.loc["平均"] = folds.mean()
folds.loc["標準差"] = folds.iloc[:5].std(ddof=0)
print(folds.to_string(float_format="%.3f"))

print("\n== 2. 不分組 vs 依 incident 分組（梯度提升）==")
rows = []
for name, splitter, g in [("StratifiedKFold（不分組）", StratifiedKFold(5, shuffle=True, random_state=0), None),
                          ("StratifiedGroupKFold", cv, groups)]:
    r = cross_validate(hgb(), X, y, groups=g, cv=splitter, scoring=SCORING)
    rows.append({"切法": name, **{m: r[f"test_{m}"].mean() for m in SCORING}})
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))

print("\n== 3. 交叉驗證版成績表（平均 ± 標準差）==")
models = {
    "邏輯迴歸": make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000)),
    "隨機森林": make_pipeline(make_preprocessor(scale=False), RandomForestClassifier(n_estimators=200, random_state=0)),
    "梯度提升": hgb(),
}
rows = []
for name, model in models.items():
    r = cross_validate(model, X, y, groups=groups, cv=cv, scoring=SCORING)
    rows.append({"model": name, **{m: f"{r[f'test_{m}'].mean():.3f} ± {r[f'test_{m}'].std():.3f}" for m in SCORING}})
print(pd.DataFrame(rows).to_string(index=False))
