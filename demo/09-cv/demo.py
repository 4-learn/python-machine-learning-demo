"""09｜scikit-learn cross_validate、StratifiedGroupKFold：交叉驗證。"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CATEGORICAL = ["event_type", "location", "source"]

df = pd.read_csv("story/data/factory/alerts.csv")
df["occurred_at"] = pd.to_datetime(df["occurred_at"])
train = df[df["occurred_at"] < "2025-12-29"]          # 前 8 週；最後 2 週的考卷不碰
X, y = train[NUMERIC + CATEGORICAL], train["label"]


def make_preprocessor(scale=True):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def logreg():
    return make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000))


def boosting():
    return make_pipeline(make_preprocessor(scale=False), HistGradientBoostingClassifier(random_state=0))


print("== 1. 只切一次：換 random_state，分數就晃 ==")
rows = []
for seed in range(5):
    X_a, X_b, y_a, y_b = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    pred = logreg().fit(X_a, y_a).predict(X_b)
    rows.append({"random_state": seed, "precision": precision_score(y_b, pred),
                 "recall": recall_score(y_b, pred)})
print(pd.DataFrame(rows).to_string(index=False, float_format="{:.3f}".format))

print("\n== 2. average_precision：不用選門檻的排序分數 ==")
truth = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
good = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]   # 3 筆真的全排在最前面
mixed = [0.9, 0.5, 0.2, 0.8, 0.7, 0.6, 0.4, 0.3, 0.1, 0.0]  # 真的被誤報夾在中間
worst = good[::-1]                                         # 真的全排在最後面
for name, score in [("全排最前", good), ("夾在中間", mixed), ("全排最後", worst)]:
    print(f"{name}：AP = {average_precision_score(truth, score):.3f}")

print("\n== 3. cross_validate：5 折，一次算三種分數 ==")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
s = cross_validate(logreg(), X, y, cv=cv, scoring=["precision", "recall", "average_precision"])
print("s 的 key：", list(s))
print(pd.DataFrame({k: v for k, v in s.items() if k.startswith("test_")}).to_string(float_format="{:.3f}".format))

print("\n== 4. 平均 ± 標準差 ==")
for key in ["test_precision", "test_recall", "test_average_precision"]:
    print(f"{key:<24}{s[key].mean():.3f} ± {s[key].std():.3f}")

print("\n== 5. groups=incident_id：同一件事不能拆到兩邊 ==")
print("每件事被記成幾筆告警：", train.groupby("incident_id").size().value_counts().sort_index().to_dict())
plain = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
tr_idx, te_idx = next(plain.split(X, y))
shared = set(train["incident_id"].iloc[tr_idx]) & set(train["incident_id"].iloc[te_idx])
print(f"StratifiedKFold      第 1 折：兩邊都出現的 incident {len(shared)} 件")
grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
tr_idx, te_idx = next(grouped.split(X, y, groups=train["incident_id"]))
shared = set(train["incident_id"].iloc[tr_idx]) & set(train["incident_id"].iloc[te_idx])
print(f"StratifiedGroupKFold 第 1 折：兩邊都出現的 incident {len(shared)} 件")

print("\n== 6. 不分組會高估多少（梯度提升，AP）==")
for name, groups, splitter in [("StratifiedKFold", None, plain),
                               ("StratifiedGroupKFold", train["incident_id"], grouped)]:
    ap = cross_validate(boosting(), X, y, groups=groups, cv=splitter, scoring="average_precision")["test_score"]
    print(f"{name:<21}{ap.mean():.3f} ± {ap.std():.3f}")
