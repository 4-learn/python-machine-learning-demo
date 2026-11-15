#@all
"""告警判斷器：替 API 課的平台決定「這筆告警要不要叫人」（ML 課進度存檔 {VERSION}）。

執行（在這個資料夾裡）：
    LINE=factory python judge.py     # 工廠線
    LINE=care python judge.py        # 健康線
"""
import json
import math
import os
from pathlib import Path

import pandas as pd
#@v04
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
#@v05
from sklearn.neighbors import KNeighborsClassifier
#@v06
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
#@v07
from sklearn.linear_model import LogisticRegression
#@v08
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#@v09
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
#@v10
from sklearn.model_selection import GridSearchCV
#@v11
from sklearn.metrics import classification_report, confusion_matrix
#@v12
from sklearn.metrics import precision_recall_curve
#@v13
from sklearn.inspection import permutation_importance
#@v14
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
#@v15
import hashlib

import joblib
#@all

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
#@v02


# 02：哪些欄位可以當特徵。事後才知道的處理紀錄不行。
NUMERIC = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CATEGORICAL = ["event_type", "location", "source"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "label"
LEAKY = ["acked_at", "resolved_at", "status"]


def xy(df):
    return df[FEATURES], df[TARGET]
#@v03


# 03：用時間切。前 6 週學（fit）、第 7–8 週比較模型（valid）、最後 2 週是考卷（test）。
VALID_START = pd.Timestamp("2025-12-15")
TEST_START = pd.Timestamp("2025-12-29")


def split_by_time(df):
    fit = df[df["occurred_at"] < VALID_START]
    valid = df[(df["occurred_at"] >= VALID_START) & (df["occurred_at"] < TEST_START)]
    test = df[df["occurred_at"] >= TEST_START]
    return fit, valid, test
#@v04


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
#@v06


def make_preprocessor(scale=True):
    """數值欄縮放（樹模型可以不縮放），類別欄 one-hot。"""
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
#@v05


def model_rows(fit, valid):
    X_fit, y_fit = xy(fit)
    X_valid, y_valid = xy(valid)
    rows = []
    knn_raw = KNeighborsClassifier(n_neighbors=15).fit(fit[NUMERIC], y_fit)
    rows.append(score_row("KNN 未縮放", y_valid, knn_raw.predict(valid[NUMERIC])))
#@v06
    knn = make_pipeline(make_preprocessor(), KNeighborsClassifier(n_neighbors=15)).fit(X_fit, y_fit)
    rows.append(score_row("KNN", y_valid, knn.predict(X_valid)))
#@v07
    logreg = make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000)).fit(X_fit, y_fit)
    rows.append(score_row("邏輯迴歸", y_valid, logreg.predict(X_valid)))
#@v08
    for name, model in candidate_models().items():
        model.fit(X_fit, y_fit)
        rows.append(score_row(name, y_valid, model.predict(X_valid)))
#@v05
    return rows
#@v08


def candidate_models():
    return {
        "決策樹": make_pipeline(make_preprocessor(scale=False), DecisionTreeClassifier(random_state=0)),
        "隨機森林": make_pipeline(make_preprocessor(scale=False),
                              RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1)),
        "梯度提升": make_pipeline(make_preprocessor(scale=False), HistGradientBoostingClassifier(random_state=0)),
    }
#@v09


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
#@v10


GRID = {"histgradientboostingclassifier__learning_rate": [0.05, 0.1],
        "histgradientboostingclassifier__max_depth": [3, None],
        "histgradientboostingclassifier__max_leaf_nodes": [15, 31]}


def tune(train):
    X, y = xy(train)
    search = GridSearchCV(candidate_models()["梯度提升"], GRID, scoring="average_precision", cv=CV, n_jobs=1)
    search.fit(X, y, groups=train["incident_id"])
    return search
#@v11


# 10 節選出的參數（兩條線各自調出來的結果，存在設定檔）
def final_model():
    params = {k.split("__")[1]: v for k, v in PROFILE["model_params"].items()}
    return make_pipeline(make_preprocessor(scale=False), HistGradientBoostingClassifier(random_state=0, **params))
#@v12


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
#@v14


def rare_alerts(train):
    """沒有標籤也能做的事：把罕見的告警挑出來給人看。"""
    X = make_preprocessor().fit_transform(train[FEATURES])
    forest = IsolationForest(n_estimators=200, contamination=0.02, random_state=0).fit(X)
    return forest.predict(X) == -1


def alert_groups(train, k=3):
    X = StandardScaler().fit_transform(train[NUMERIC])
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
#@v15


# ---------------------------------------------------------------- Workshop 1：判斷器 MVP
MODEL_DIR = HERE / "out" / LINE


def train_and_save():
    """用前 8 週訓練定版模型，門檻在 valid 上選，存成 joblib＋JSON。"""
    df = load_alerts()
    fit, valid, test = split_by_time(df)
    probe = final_model().fit(*xy(fit))
    high, low = choose_thresholds(valid[TARGET], probe.predict_proba(xy(valid)[0])[:, 1])
    model = final_model().fit(*xy(pd.concat([fit, valid])))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "judge.joblib")
    config = {"line": LINE, "high": high, "low": low,
              "trained_until": str(TEST_START.date()), "sha256_12": sha256_12(MODEL_DIR / "judge.joblib")}
    (MODEL_DIR / "judge.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    return config


def sha256_12(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


class Judge:
    """平台收到一筆事件時呼叫：回傳 auto_alert／require_review／log_only。"""

    def __init__(self, model_dir=MODEL_DIR):
        self.config = json.loads((model_dir / "judge.json").read_text(encoding="utf-8"))
        path = model_dir / "judge.joblib"
        if sha256_12(path) != self.config["sha256_12"]:
            raise RuntimeError("模型檔和設定檔對不上，拒絕啟動")
        self.model = joblib.load(path)
        people = pd.read_csv(STORY / "data" / LINE / "people.csv")
        self.days_on_site = dict(zip(people["person_id"], people["days_on_site"]))
        history = load_alerts()
        self.history = history[history[TARGET] == 1][["person_id", "occurred_at"]]

    def features(self, event):
        """把 API 課格式的事件，補成模型要的一列特徵。"""
        t = pd.Timestamp(event["occurred_at"])
        pid = event["detail"]["person_id"]
        past = self.history[(self.history["person_id"] == pid) & (self.history["occurred_at"] < t)
                            & (self.history["occurred_at"] >= t - pd.Timedelta(days=30))]
        return pd.DataFrame([{
            "confidence": event["confidence"],
            "duration_seconds": event["detail"]["duration_seconds"],
            "light_lux": event["detail"]["light_lux"],
            "days_on_site": self.days_on_site.get(pid, 0),
            "prior_alerts_30d": len(past),
            "hour": t.hour,
            "event_type": event["event_type"], "location": event["location"], "source": event["source"],
        }])[FEATURES]

    def decide(self, event):
        score = float(self.model.predict_proba(self.features(event))[0, 1])
        c = self.config
        decision = "auto_alert" if score >= c["high"] else "require_review" if score >= c["low"] else "log_only"
        return {"decision": decision, "score": round(score, 3), "model": c["sha256_12"]}
#@all


def main():
    df = load_alerts()
#@main v01
    df["decision"] = platform_rule(df)
    print(f"== {LINE}：平台試用 10 週的告警 ==")
    print(f"{len(df)} 筆告警，{df['occurred_at'].dt.date.nunique()} 天；處理結果：")
    print(df["status"].value_counts().to_string())
    print("\n== 平台規則的決定 × 值班人員最後的處理 ==")
    print(pd.crosstab(df["decision"], df["status"], margins=True, margins_name="合計").to_string())
    auto = df[df["decision"] == "auto_alert"]
    print(f"\n自動叫人 {len(auto)} 次，其中誤報 {(auto['status'] == 'dismissed').sum()} 次"
          f"（{(auto['status'] == 'dismissed').mean():.1%}）")
    missed = df[(df["decision"] == "log_only") & (df["status"] == "resolved")]
    print(f"只記錄、沒叫人的告警裡，事後確認是真的：{len(missed)} 筆")
#@main v02
    X, y = xy(df)
    print(f"== {LINE}：告警歷史變成 X 和 y ==")
    print(f"X：{X.shape[0]} 列 × {X.shape[1]} 欄；y：要叫人的比例 {y.mean():.3f}")
    print("數值特徵：", ", ".join(NUMERIC))
    print("類別特徵：", ", ".join(CATEGORICAL))
    print("不能當特徵（事後才知道）：", ", ".join(LEAKY))
    print("\n== 為什麼 resolved_at 不能用 ==")
    print(pd.crosstab(df["resolved_at"].notna().map({True: "有結案時間", False: "沒有"}), y).to_string())
#@main v03
    fit, valid, test = split_by_time(df)
    print(f"== {LINE}：用時間切出三份 ==")
    for name, part in [("fit", fit), ("valid", valid), ("test", test)]:
        print(f"{name:<6}{len(part):>5} 筆  {part['occurred_at'].min():%m-%d}～{part['occurred_at'].max():%m-%d}"
              f"  要叫人 {part[TARGET].mean():.3f}")
    shared = set(fit["incident_id"]) & (set(valid["incident_id"]) | set(test["incident_id"]))
    print(f"同一件事被切到兩邊：{len(shared)} 件")
#@main v04
    fit, valid, _ = split_by_time(df)
    print(f"== {LINE}：成績表（valid）==")
    print(pd.DataFrame(baseline_rows(fit, valid)).to_string(index=False))
#@main v05 v06 v07 v08
    fit, valid, _ = split_by_time(df)
    print(f"== {LINE}：成績表（valid）==")
    print(pd.DataFrame(baseline_rows(fit, valid) + model_rows(fit, valid)).to_string(index=False))
#@main v07
    logreg = make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000)).fit(*xy(fit))
    coef = pd.Series(logreg[-1].coef_[0], index=logreg[0].get_feature_names_out())
    coef = coef.rename(lambda name: name.split("__")[1])
    print("\n== 邏輯迴歸的係數：正的往「要叫人」推、負的往「不用」推 ==")
    print(coef.sort_values(ascending=False).round(2).to_string())
#@main v09
    fit, valid, _ = split_by_time(df)
    train = pd.concat([fit, valid])
    print(f"== {LINE}：5 折交叉驗證（依 incident 分組，前 8 週）==")
    print(cv_table(train).to_string(index=False))
#@main v10
    fit, valid, test = split_by_time(df)
    train = pd.concat([fit, valid])
    search = tune(train)
    best = {k.split("__")[1]: v for k, v in search.best_params_.items()}
    print(f"== {LINE}：GridSearchCV（{len(search.cv_results_['params'])} 組 × 5 折）==")
    print(f"最佳參數：{best}；CV average precision {search.best_score_:.3f}")
    X_test, y_test = xy(test)
    rows = [score_row("平台規則 conf>=0.8", y_test, (platform_rule(test) == "auto_alert").astype(int)),
            score_row("梯度提升（調參後）", y_test, search.best_estimator_.predict(X_test))]
    print("\n== 打開考卷（test，只開這一次）==")
    print(pd.DataFrame(rows).to_string(index=False))
#@main v11
    fit, valid, _ = split_by_time(df)
    model = final_model().fit(*xy(fit))
    X_valid, y_valid = xy(valid)
    pred = model.predict(X_valid)
    print(f"== {LINE}：混淆矩陣（valid，門檻 0.5）==")
    tn, fp, fn, tp = confusion_matrix(y_valid, pred).ravel()
    print(f"不用叫、沒叫 {tn}｜不用叫、卻叫了（誤報）{fp}")
    print(f"要叫、沒叫（漏報）{fn}｜要叫、叫了 {tp}")
    print("\n" + classification_report(y_valid, pred, target_names=["不用叫", "要叫人"], digits=3))
#@main v12 v13
    fit, valid, _ = split_by_time(df)
    model = final_model().fit(*xy(fit))
    X_valid, y_valid = xy(valid)
    proba = model.predict_proba(X_valid)[:, 1]
    high, low = choose_thresholds(y_valid, proba)
#@main v12
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
#@main v13
    decision = decide_by_score(proba, high, low)
    imp = permutation_importance(model, X_valid, y_valid, scoring="average_precision", n_repeats=5, random_state=0)
    print(f"== {LINE}：打亂一欄，分數掉多少（permutation importance）==")
    print(pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).round(3).to_string())
    errors = valid.assign(decision=decision.astype(str))
    missed = errors[(errors["decision"] == "log_only") & (errors[TARGET] == 1)]
    false_auto = errors[(errors["decision"] == "auto_alert") & (errors[TARGET] == 0)]
    print(f"\n== 錯誤清單：漏報 {len(missed)} 筆、自動叫人卻是誤報 {len(false_auto)} 筆 ==")
    cols = ["alert_id", "event_type", "location", "confidence", "duration_seconds", "light_lux"]
    print(pd.concat([missed.assign(錯誤="漏報"), false_auto.assign(錯誤="誤報")])[["錯誤"] + cols].to_string(index=False))
#@main v14
    fit, valid, _ = split_by_time(df)
    train = pd.concat([fit, valid])
    groups = alert_groups(train)
    print(f"== {LINE}：KMeans 把告警分成 3 群（沒看標籤）==")
    print(train.assign(群=groups).groupby("群").agg(告警數=(TARGET, "size"), 平均信心值=("confidence", "mean"),
                                                 平均秒數=("duration_seconds", "mean"), 事後是真的=(TARGET, "mean")).round(3).to_string())
    rare = rare_alerts(train)
    print(f"\n== IsolationForest 挑出最罕見的 {rare.sum()} 筆，交給人看 ==")
    print(train[rare][["alert_id", "event_type", "location", "confidence", "duration_seconds", "light_lux", "hour"]]
          .head(8).to_string(index=False))
#@main v15
    config = train_and_save()
    print(f"== {LINE}：定版判斷器 ==")
    print(f"門檻 HIGH {config['high']}、LOW {config['low']}；模型 {config['sha256_12']}（訓練到 {config['trained_until']} 前）")
    judge = Judge()
    print("\n== 最後 2 週挑 8 筆新事件（API 課格式），交給 decide() ==")
    rows = []
    for line in (STORY / "data" / LINE / "new_events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        result = judge.decide(event)
        rows.append({"occurred_at": event["occurred_at"][5:16], "event_type": event["event_type"],
                     "location": event["location"], "confidence": event["confidence"],
                     "平台規則": str(platform_rule(pd.DataFrame([event])).iloc[0]),
                     "判斷器": result["decision"], "score": result["score"], "事後": event["_answer"]})
    print(pd.DataFrame(rows).to_string(index=False))
#@all


if __name__ == "__main__":
    main()
