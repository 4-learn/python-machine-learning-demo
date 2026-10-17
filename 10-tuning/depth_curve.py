"""10｜驗證曲線：決策樹越深，訓練分數一路上升，驗證分數卻先升後降。"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from common import load_events, make_preprocessor, time_split, xy

train, _ = time_split(load_events())  # 只用前 8 週；test 不碰
X, y = xy(train)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
tree = make_pipeline(make_preprocessor(scale=False), DecisionTreeClassifier(random_state=0))
depths = [1, 2, 3, 4, 5, 6, 8, 10, 15, None]

curve = pd.DataFrame({"max_depth": [str(d) for d in depths]})
for scoring, label in [("accuracy", "acc"), ("average_precision", "AP")]:
    tr, va = validation_curve(tree, X, y, param_name="decisiontreeclassifier__max_depth", param_range=depths,
                              groups=train["incident_id"], cv=cv, scoring=scoring)
    curve[f"{label} 訓練"] = tr.mean(axis=1)
    curve[f"{label} 驗證"] = va.mean(axis=1)
curve["AP 差距"] = curve["AP 訓練"] - curve["AP 驗證"]

print("== 1. 決策樹深度 vs 訓練／驗證分數（5 折平均）==")
print(curve.to_string(index=False, float_format="%.3f"))
best = curve.loc[curve["AP 驗證"].idxmax()]
print(f"驗證 AP 最高：max_depth={best['max_depth']}（{best['AP 驗證']:.3f}）")
