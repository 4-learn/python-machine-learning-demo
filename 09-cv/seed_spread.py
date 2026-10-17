"""09｜同一個模型、同一份資料，只換切割的 random_state，分數會差多少？"""
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

from common import load_events, make_preprocessor, score_row, time_split, xy

threadpool_limits(2)  # 梯度提升預設吃滿所有核心；學員機核心少、又和別的程式搶，限 2 條反而快

train, _ = time_split(load_events())
X, y = xy(train)
groups = train["incident_id"]

models = {
    "邏輯迴歸": lambda: make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000)),
    "隨機森林": lambda: make_pipeline(make_preprocessor(scale=False), RandomForestClassifier(n_estimators=200, random_state=0)),
    "梯度提升": lambda: make_pipeline(make_preprocessor(scale=False), HistGradientBoostingClassifier(random_state=0)),
}

rows = []
for seed in range(10):
    # 依 incident 分組、隨機抽 25% 當驗證；唯一改變的是 random_state
    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
    row = {"seed": seed, "驗證正例": int(y.iloc[val_idx].sum())}
    for name, make in models.items():
        m = make().fit(X.iloc[fit_idx], y.iloc[fit_idx])
        row[name] = score_row(name, y.iloc[val_idx], m.predict(X.iloc[val_idx]))["recall"]
    row["第一名"] = max(models, key=lambda n: row[n])
    rows.append(row)
table = pd.DataFrame(rows)

print("== 1. 梯度提升：換 10 次 random_state 的 recall ==")
hgb = table["梯度提升"]
print("recall：" + " ".join(f"{v:.3f}" for v in hgb))
print(f"最低 {hgb.min():.3f}、最高 {hgb.max():.3f}，差 {hgb.max() - hgb.min():.3f}")
print(f"每次驗證集約 {table['驗證正例'].mean():.0f} 個該通知的事件：多抓或少抓 1 個，recall 就變 {1 / table['驗證正例'].mean():.3f}")

print("\n== 2. 三個模型比 recall，誰是第一名？ ==")
print(table.to_string(index=False, float_format="%.3f"))
print(table["第一名"].value_counts().to_string())
