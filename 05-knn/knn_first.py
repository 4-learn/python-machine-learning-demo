"""05｜第一個模型：KNN（未縮放）。模型的一生：建立 → fit → predict → 和 baseline 比。"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier

from common import NUMERIC, TARGET, current_rule, fit_valid_split, load_events, score_row, time_split

train, _ = time_split(load_events())          # test 留到 10 節，這裡碰都不碰
fit, valid = fit_valid_split(train)
X_fit, y_fit = fit[NUMERIC], fit[TARGET]       # KNN 只吃數字：今天先用 7 個數值欄
X_valid, y_valid = valid[NUMERIC], valid[TARGET]

print("== 1. fit 用來學、valid 用來考 ==")
print(f"fit   {len(fit)} 筆（{fit.event_time.min():%m-%d}～{fit.event_time.max():%m-%d}），該通知 {y_fit.mean():.3f}")
print(f"valid {len(valid)} 筆（{valid.event_time.min():%m-%d}～{valid.event_time.max():%m-%d}），該通知 {y_valid.mean():.3f}")
print("特徵：", ", ".join(NUMERIC))

print("\n== 2. 模型建好了，但還什麼都沒學 ==")
knn = KNeighborsClassifier(n_neighbors=15)
print(knn)
try:
    knn.predict(X_valid)
except NotFittedError as e:
    print("還沒 fit 就 predict →", type(e).__name__)

print("\n== 3. fit：把訓練資料存起來 ==")
knn.fit(X_fit, y_fit)
print(f"記住了 {knn.n_samples_fit_} 筆、{knn.n_features_in_} 個欄位；答案只有這幾種：{knn.classes_.tolist()}")

print("\n== 4. predict：只看 X，不看答案 ==")
y_pred = knn.predict(X_valid)
print(f"y_pred 型態 {type(y_pred).__name__}，長度 {len(y_pred)}；y_valid 長度 {len(y_valid)}")
look = pd.DataFrame({"event_id": valid.event_id.values[:8], "預測": y_pred[:8], "答案": y_valid.values[:8]})
print(look.to_string(index=False))

print("\n== 5. 看附近的人是誰 ==")
one = valid[valid[TARGET] == 1].iloc[[0]]
print(one[["event_id", "zone"] + NUMERIC].to_string(index=False))
dist, idx = knn.kneighbors(one[NUMERIC])
near = fit.iloc[idx[0]][["event_id", "zone"] + NUMERIC + [TARGET]].copy()
near.insert(1, "距離", dist[0].round(1))
print(near.head(8).to_string(index=False))
votes = int(near[TARGET].sum())
print(f"15 個鄰居裡 {votes} 個該通知、{15 - votes} 個不必 → 預測 {int(knn.predict(one[NUMERIC])[0])}，答案 {int(one[TARGET].iloc[0])}")

print("\n== 6. 成績表（valid）==")
dummy = DummyClassifier(strategy="most_frequent").fit(X_fit, y_fit)
table = pd.DataFrame([
    score_row("多數類", y_valid, dummy.predict(X_valid)),
    score_row("現行規則 conf>=0.7", y_valid, current_rule(valid)),
    score_row("KNN k=15 未縮放", y_valid, y_pred),
])
print(table.to_string(index=False))

print("\n== 7. 考自己做過的題目 ==")
knn1 = KNeighborsClassifier(n_neighbors=1).fit(X_fit, y_fit)
print(f"KNN k=1  在 fit 上 accuracy {knn1.score(X_fit, y_fit):.3f}；在 valid 上 {knn1.score(X_valid, y_valid):.3f}")
