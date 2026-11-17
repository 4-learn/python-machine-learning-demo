"""07｜scikit-learn LogisticRegression：輸出機率與係數。"""
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["confidence", "duration_seconds", "light_lux", "days_on_site", "prior_alerts_30d", "hour"]
CAT = ["event_type", "location", "source"]
df = pd.read_csv("story/data/factory/alerts.csv", parse_dates=["occurred_at"])
fit = df[df["occurred_at"] < "2025-12-15"]
valid = df[(df["occurred_at"] >= "2025-12-15") & (df["occurred_at"] < "2025-12-29")]

print("== 1. predict_proba：每筆兩個機率 ==")
sample = valid.iloc[[0, 79, 69, 5]]
pre = ColumnTransformer([("num", StandardScaler(), NUM),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
model = make_pipeline(pre, LogisticRegression(max_iter=1000)).fit(fit[NUM + CAT], fit["label"])
proba = model.predict_proba(sample[NUM + CAT])
print(pd.DataFrame({"confidence": sample["confidence"].values,
                    "duration_seconds": sample["duration_seconds"].values,
                    "P(不用叫)": proba[:, 0].round(3), "P(要叫人)": proba[:, 1].round(3),
                    "predict": model.predict(sample[NUM + CAT]), "label": sample["label"].values},
                   index=sample["alert_id"]).to_string())
print("classes_：", model.classes_.tolist())

print("\n== 2. predict 其實就是 P(要叫人) >= 0.5 ==")
p = model.predict_proba(valid[NUM + CAT])[:, 1]
same = (model.predict(valid[NUM + CAT]) == (p >= 0.5).astype(int)).mean()
print(f"valid {len(valid)} 筆，兩種寫法相同的比例：{same:.3f}")
for t in [0.3, 0.5, 0.7]:
    print(f"門檻 {t}：叫人 {(p >= t).sum()} 次")

print("\n== 3. 機率怎麼分布 ==")
print(pd.cut(pd.Series(p), bins=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]).value_counts(sort=False).to_string())

print("\n== 4. coef_ 配上 get_feature_names_out ==")
coef = pd.Series(model[-1].coef_[0], index=model[0].get_feature_names_out())
print(coef.sort_values(ascending=False).round(3).head(5).to_string())
print("...")
print(coef.sort_values(ascending=False).round(3).tail(3).to_string())
print(f"intercept_：{model[-1].intercept_[0]:.3f}")

print("\n== 5. 不縮放的係數：單位不同，大小不能比 ==")
raw = LogisticRegression(max_iter=1000).fit(fit[NUM], fit["label"])
scaled = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(fit[NUM], fit["label"])
print(pd.DataFrame({"未縮放": raw.coef_[0], "縮放後": scaled[-1].coef_[0]}, index=NUM).round(3).to_string())

print("\n== 6. max_iter：還沒算完就停 ==")
for n in [10, 1000]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = LogisticRegression(max_iter=n).fit(fit[NUM], fit["label"])
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    print(f"max_iter={n}：實際跑了 {m.n_iter_[0]} 輪，ConvergenceWarning：{'有' if warned else '沒有'}")

