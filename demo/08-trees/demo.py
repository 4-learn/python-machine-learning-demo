"""08｜scikit-learn 決策樹與集成：DecisionTreeClassifier、RandomForestClassifier、HistGradientBoostingClassifier。"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CATEGORICAL = ["event_type", "location", "source"]

df = pd.read_csv("story/data/factory/alerts.csv")
df["occurred_at"] = pd.to_datetime(df["occurred_at"])
fit = df[df["occurred_at"] < "2025-12-15"]
valid = df[(df["occurred_at"] >= "2025-12-15") & (df["occurred_at"] < "2025-12-29")]
y_fit, y_valid = fit["label"], valid["label"]


def make_preprocessor(scale=True):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def scores(name, pred):
    return {"model": name, "precision": round(precision_score(y_valid, pred), 3),
            "recall": round(recall_score(y_valid, pred), 3), "alerts": int(pred.sum())}


print("== 1. export_text：深度 2 的樹學到的規則（工廠線，前 6 週）==")
tree2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(fit[NUMERIC], y_fit)
print(export_text(tree2, feature_names=NUMERIC, decimals=3))

print("== 2. 和平台規則比（valid）==")
rule = (valid["confidence"] >= 0.8).astype(int)
rows = [scores("平台規則 conf>=0.8", rule), scores("深度 2 的樹", tree2.predict(valid[NUMERIC]))]
print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))

print("\n== 3. max_depth：越深越會背答案 ==")
rows = []
for depth in [1, 2, 3, 4, 6, 8, None]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(fit[NUMERIC], y_fit)
    rows.append({"max_depth": str(depth), "leaves": t.get_n_leaves(),
                 "fit_acc": round(accuracy_score(y_fit, t.predict(fit[NUMERIC])), 3),
                 "valid_acc": round(accuracy_score(y_valid, t.predict(valid[NUMERIC])), 3)})
print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))

print("\n== 4. 一棵樹 vs 一群樹：背答案的程度 ==")
X_fit, X_valid = fit[NUMERIC + CATEGORICAL], valid[NUMERIC + CATEGORICAL]
models = {"決策樹（不限深度）": DecisionTreeClassifier(random_state=0),
          "隨機森林 300 棵": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1),
          "梯度提升": HistGradientBoostingClassifier(random_state=0)}
rows = []
for name, model in models.items():
    m = make_pipeline(make_preprocessor(scale=False), model).fit(X_fit, y_fit)
    rows.append({"model": name, "fit_acc": round(accuracy_score(y_fit, m.predict(X_fit)), 3),
                 "valid_acc": round(accuracy_score(y_valid, m.predict(X_valid)), 3)})
print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))
