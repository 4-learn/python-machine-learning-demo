"""告警判斷器：替 API 課的平台決定「這筆告警要不要叫人」（ML 課進度存檔 v12）。

執行（在這個資料夾裡）：
    LINE=factory python judge.py     # 工廠線
    LINE=care python judge.py        # 健康線
"""
import json
import math
import os
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve

LINE = os.getenv("LINE", "factory")
HERE = Path(__file__).resolve().parent
STORY = HERE.parent
PROFILE = json.loads((STORY / "profiles" / f"{LINE}.json").read_text(encoding="utf-8"))
DATA = STORY / "data" / LINE / "alerts.csv"


def load_alerts():
    """讀入平台 10 週的告警歷史。"""
    df = pd.read_csv(DATA)
    df["occurred_at"] = pd.to_datetime(df["occurred_at"])
    return df


def platform_rule(df):
    """API 課 v0.12 的規則：信心值 0.8 以上自動開告警、0.5 到 0.8 待人確認、其餘只記錄。"""
    return pd.cut(df["confidence"], bins=[0, 0.5, 0.8, 1.01], right=False,
                  labels=["log_only", "require_review", "auto_alert"])


# 02：哪些欄位可以當特徵。事後才知道的處理紀錄不行。
NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CATEGORICAL = ["event_type", "location", "source"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "label"
LEAKY = ["acked_at", "resolved_at", "status"]


def xy(df):
    return df[FEATURES], df[TARGET]


# 03：用時間切。前 6 週學（fit）、第 7–8 週比較模型（valid）、最後 2 週是考卷（test）。
VALID_START = pd.Timestamp("2025-12-15")
TEST_START = pd.Timestamp("2025-12-29")


def split_by_time(df):
    fit = df[df["occurred_at"] < VALID_START]
    valid = df[(df["occurred_at"] >= VALID_START) & (df["occurred_at"] < TEST_START)]
    test = df[df["occurred_at"] >= TEST_START]
    return fit, valid, test


def score_row(name, y_true, y_pred):
    """成績表的一列：叫人的次數，以及叫得準不準、漏了多少。"""
    return {"model": name,
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred), 3),
            "alerts": int(pd.Series(y_pred).sum())}


def baseline_rows(fit, valid):
    X_fit, y_fit = xy(fit)
    X_valid, y_valid = xy(valid)
    dummy = DummyClassifier().fit(X_fit, y_fit)
    rule = (platform_rule(valid) == "auto_alert").astype(int)
    return [score_row("多數類", y_valid, dummy.predict(X_valid)),
            score_row("平台規則 conf>=0.8", y_valid, rule)]


def make_preprocessor(scale=True):
    """數值欄縮放（樹模型可以不縮放），類別欄 one-hot。"""
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def model_rows(fit, valid):
    X_fit, y_fit = xy(fit)
    X_valid, y_valid = xy(valid)
    rows = []
    knn_raw = KNeighborsClassifier(n_neighbors=15).fit(fit[NUMERIC], y_fit)
    rows.append(score_row("KNN 未縮放", y_valid, knn_raw.predict(valid[NUMERIC])))
    knn = make_pipeline(make_preprocessor(), KNeighborsClassifier(n_neighbors=15)).fit(X_fit, y_fit)
    rows.append(score_row("KNN", y_valid, knn.predict(X_valid)))
    logreg = make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000)).fit(X_fit, y_fit)
    rows.append(score_row("邏輯迴歸", y_valid, logreg.predict(X_valid)))
    for name, model in candidate_models().items():
        model.fit(X_fit, y_fit)
        rows.append(score_row(name, y_valid, model.predict(X_valid)))
    return rows


def candidate_models():
    return {
        "決策樹": make_pipeline(make_preprocessor(scale=False), DecisionTreeClassifier(random_state=0)),
        "隨機森林": make_pipeline(make_preprocessor(scale=False),
                              RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1)),
        "梯度提升": make_pipeline(make_preprocessor(scale=False), HistGradientBoostingClassifier(random_state=0)),
    }


CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)


def cv_table(train):
    X, y = xy(train)
    models = {"邏輯迴歸": make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000)),
              "隨機森林": candidate_models()["隨機森林"],
              "梯度提升": candidate_models()["梯度提升"]}
    rows = []
    for name, model in models.items():
        s = cross_validate(model, X, y, groups=train["incident_id"], cv=CV,
                           scoring=["precision", "recall", "average_precision"])
        rows.append({"model": name,
                     "precision": f"{s['test_precision'].mean():.3f} ± {s['test_precision'].std():.3f}",
                     "recall": f"{s['test_recall'].mean():.3f} ± {s['test_recall'].std():.3f}",
                     "AP": f"{s['test_average_precision'].mean():.3f} ± {s['test_average_precision'].std():.3f}"})
    return pd.DataFrame(rows)


GRID = {"histgradientboostingclassifier__learning_rate": [0.05, 0.1],
        "histgradientboostingclassifier__max_depth": [3, None],
        "histgradientboostingclassifier__max_leaf_nodes": [15, 31]}


def tune(train):
    X, y = xy(train)
    search = GridSearchCV(candidate_models()["梯度提升"], GRID, scoring="average_precision", cv=CV, n_jobs=1)
    search.fit(X, y, groups=train["incident_id"])
    return search


# 10 節選出的參數（兩條線各自調出來的結果，存在設定檔）
def final_model():
    params = {k.split("__")[1]: v for k, v in PROFILE["model_params"].items()}
    return make_pipeline(make_preprocessor(scale=False), HistGradientBoostingClassifier(random_state=0, **params))


def choose_thresholds(y_true, proba):
    """HIGH：自動開告警時，真的至少要占 auto_alert_min_precision。
    LOW：低於它才只記錄，漏掉的真事件不超過 max_missed_rate。兩個數字都在設定檔。"""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    high = thresholds[(precision >= PROFILE["auto_alert_min_precision"]).argmax()]
    low = thresholds[(recall >= 1 - PROFILE["max_missed_rate"]).nonzero()[0].max()]
    # 只留 3 位小數時一律無條件捨去：門檻變低一點，只會多叫、不會多漏
    return math.floor(high * 1000) / 1000, math.floor(low * 1000) / 1000


def decide_by_score(proba, high, low):
    return pd.cut(proba, bins=[-0.01, low, high, 1.01], right=False,
                  labels=["log_only", "require_review", "auto_alert"])


def band_table(decision, y_true):
    t = pd.DataFrame({"decision": decision, "label": y_true.to_numpy()})
    out = t.groupby("decision", observed=False)["label"].agg(["count", "sum"])
    out.columns = ["告警數", "其中真的"]
    return out.reindex(["auto_alert", "require_review", "log_only"])


def main():
    df = load_alerts()
    fit, valid, _ = split_by_time(df)
    model = final_model().fit(*xy(fit))
    X_valid, y_valid = xy(valid)
    proba = model.predict_proba(X_valid)[:, 1]
    high, low = choose_thresholds(y_valid, proba)
    print(f"== {LINE}：兩個門檻（valid 上選）==")
    print(f"HIGH = {high}（自動叫人時至少 {PROFILE['auto_alert_min_precision']:.0%} 是真的）")
    print(f"LOW  = {low}（只記錄的告警裡，漏掉的真事件不超過 {PROFILE['max_missed_rate']:.0%}）")
    print("\n平台規則（信心值 0.8／0.5）：")
    print(band_table(platform_rule(valid), y_valid).to_string())
    print(f"\n判斷器（模型分數 {high}／{low}）：")
    print(band_table(decide_by_score(proba, high, low), y_valid).to_string())
    out = HERE / "out" / LINE
    out.mkdir(parents=True, exist_ok=True)
    (out / "thresholds.json").write_text(json.dumps({"high": high, "low": low}, indent=1))
    print(f"\n已寫出 out/{LINE}/thresholds.json")


if __name__ == "__main__":
    main()
