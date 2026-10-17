"""08｜決策樹：讓模型自己「寫規則」，印出來和公司的 conf >= 0.7 比。"""
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from common import current_rule, fit_valid_split, load_events, make_preprocessor, score_row, time_split, xy

train, _ = time_split(load_events())
fit, valid = fit_valid_split(train)
X_fit, y_fit = xy(fit)
X_valid, y_valid = xy(valid)

print("== 1. 不設限的決策樹 ==")
tree = make_pipeline(make_preprocessor(scale=False), DecisionTreeClassifier(random_state=0))
tree.fit(X_fit, y_fit)
print(f"深度 {tree[-1].get_depth()} 層、葉子 {tree[-1].get_n_leaves()} 片")
print(f"fit 上 accuracy {(tree.predict(X_fit) == y_fit).mean():.3f}；valid 上 {(tree.predict(X_valid) == y_valid).mean():.3f}")

print("\n== 2. 只長 3 層的樹，印出規則 ==")
small = make_pipeline(make_preprocessor(scale=False), DecisionTreeClassifier(max_depth=3, random_state=0))
small.fit(X_fit, y_fit)
names = [n.split("__")[1] for n in small[0].get_feature_names_out()]  # cat__zone_休息區 -> zone_休息區
print(export_text(small[-1], feature_names=names, decimals=2, show_weights=True))

print("== 3. 3 層樹 vs 現行規則（valid）==")
rows = [score_row("現行規則 conf>=0.7", y_valid, current_rule(valid)),
        score_row("3 層決策樹", y_valid, small.predict(X_valid)),
        score_row("不設限決策樹", y_valid, tree.predict(X_valid))]
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))
