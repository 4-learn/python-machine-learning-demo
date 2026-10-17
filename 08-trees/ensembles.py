"""08｜一棵樹 vs 一片森林 vs 一棵接一棵修正：決策樹、隨機森林、梯度提升。"""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from common import current_rule, fit_valid_split, load_events, make_preprocessor, score_row, time_split, xy

threadpool_limits(2)  # 梯度提升預設吃滿所有核心；學員機限 2 條比較穩

train, _ = time_split(load_events())
fit, valid = fit_valid_split(train)
X_fit, y_fit = xy(fit)
X_valid, y_valid = xy(valid)


def tree_model(model):
    # 樹只問「大於還是小於」，縮放不改變答案，所以 scale=False
    return make_pipeline(make_preprocessor(scale=False), model)


print("== 1. 換一批訓練資料，模型穩不穩？（fit 重抽 10 次）==")
rng = np.random.RandomState(0)
spread = []
for name, make in [("決策樹", lambda: DecisionTreeClassifier(random_state=0)),
                   ("隨機森林", lambda: RandomForestClassifier(n_estimators=200, random_state=0))]:
    recalls = []
    for _ in range(10):
        idx = rng.choice(len(X_fit), len(X_fit), replace=True)
        m = tree_model(make()).fit(X_fit.iloc[idx], y_fit.iloc[idx])
        recalls.append(score_row(name, y_valid, m.predict(X_valid))["recall"])
    spread.append({"model": name, "recall 最低": min(recalls), "recall 最高": max(recalls)})
print(pd.DataFrame(spread).to_string(index=False, float_format="%.3f"))

print("\n== 2. 成績表（valid，第 7–8 週）==")
models = [
    ("多數類", DummyClassifier()),
    ("KNN k=15 未縮放", make_pipeline(make_preprocessor(scale=False), KNeighborsClassifier(n_neighbors=15))),
    ("KNN k=15", make_pipeline(make_preprocessor(), KNeighborsClassifier(n_neighbors=15))),
    ("邏輯迴歸", make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000))),
    ("決策樹", tree_model(DecisionTreeClassifier(random_state=0))),
    ("隨機森林", tree_model(RandomForestClassifier(n_estimators=200, random_state=0))),
    ("梯度提升", tree_model(HistGradientBoostingClassifier(random_state=0))),
]
rows = []
for name, model in models:
    model.fit(X_fit, y_fit)
    rows.append(score_row(name, y_valid, model.predict(X_valid)))
rows.insert(1, score_row("現行規則 conf>=0.7", y_valid, current_rule(valid)))
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))

print("\n== 3. 有一欄缺值時 ==")
X_hole = X_valid.copy()
X_hole.loc[X_hole.sample(frac=0.2, random_state=0).index, "light_lux"] = np.nan
for name, model in models[3:]:
    try:
        model.predict(X_hole)
        print(f"{name}：可以預測")
    except ValueError:
        print(f"{name}：ValueError（不接受缺值）")
