"""06｜讓 AI 寫，然後挑錯：切割前 fit_transform、在新事件上再 fit 一次。"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import CATEGORICAL, FEATURES, NUMERIC, TARGET, fit_valid_split, load_events, make_preprocessor, score_row, time_split

# ---- AI 寫的版本（講義第 6 小節），原樣照抄 ----
df = load_events()
X = pd.get_dummies(df[NUMERIC + CATEGORICAL])
X[NUMERIC] = StandardScaler().fit_transform(X[NUMERIC])
df = pd.concat([df[["event_time", TARGET]], X], axis=1)
train, test = time_split(df)
fit, valid = fit_valid_split(train)
knn = KNeighborsClassifier(n_neighbors=15).fit(fit[X.columns], fit[TARGET])
print("== 0. AI 版本的輸出 ==")
print("valid accuracy:", knn.score(valid[X.columns], valid[TARGET]))
row_ai = score_row("AI：切割前處理", valid[TARGET], knn.predict(valid[X.columns]))

# ---- 正確寫法：先切，前處理放進 Pipeline，只從 fit 學 ----
events = load_events()
train, _ = time_split(events)
fit, valid = fit_valid_split(train)
model = Pipeline([("prep", make_preprocessor()), ("knn", KNeighborsClassifier(n_neighbors=15))])
model.fit(fit[FEATURES], fit[TARGET])
row_ok = score_row("Pipeline", valid[TARGET], model.predict(valid[FEATURES]))

print("\n== 1. 兩種寫法的成績（valid）==")
print(pd.DataFrame([row_ai, row_ok]).to_string(index=False))

print("\n== 2. 兩種寫法學到的平均值不一樣 ==")
means = pd.DataFrame({"只用 fit": fit[NUMERIC].mean(), "全部 10 週": events[NUMERIC].mean()}).round(3)
print(means.loc[["confidence", "prior_violations_30d", "tenure_days"]].to_string())

print("\n== 3. 另一種常見錯：上線時對新事件 fit_transform ==")
one = valid[valid.event_id == "E02916"][NUMERIC]
scaler = model["prep"].named_transformers_["num"]
print("StandardScaler().fit_transform(一筆新事件) →", StandardScaler().fit_transform(one).round(3).tolist()[0])
print("已 fit 的縮放器 .transform(一筆新事件)     →", scaler.transform(one).round(3).tolist()[0])
