"""04｜scikit-learn DummyClassifier：不用模型能拿幾分。順便認識 fit／predict 與三個分數。"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("== 1. 10 筆告警：fit 學、predict 答 ==")
X = pd.DataFrame({"confidence": [0.95, 0.83, 0.81, 0.72, 0.66, 0.55, 0.49, 0.41, 0.35, 0.92]})
y = pd.Series([1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X, y)
print("fit 之後記住的：classes_ =", dummy.classes_.tolist(), "，class_prior_ =", dummy.class_prior_.tolist())
print("predict：", dummy.predict(X).tolist())

print("\n== 2. 三個分數 ==")
rule = (X["confidence"] >= 0.8).astype(int)
for name, pred in [("多數類", dummy.predict(X)), ("conf>=0.8", rule)]:
    print(f"{name:<10} accuracy {accuracy_score(y, pred):.3f}  "
          f"precision {precision_score(y, pred, zero_division=0):.3f}  recall {recall_score(y, pred):.3f}")

print("\n== 3. 換成工廠線：前 6 週 fit、第 7–8 週 predict ==")
df = pd.read_csv("story/data/factory/alerts.csv")
df["occurred_at"] = pd.to_datetime(df["occurred_at"])
fit = df[df["occurred_at"] < "2025-12-15"]
valid = df[(df["occurred_at"] >= "2025-12-15") & (df["occurred_at"] < "2025-12-29")]
X_fit, y_fit = fit[["confidence"]], fit["label"]
X_valid, y_valid = valid[["confidence"]], valid["label"]

rows = []
for strategy in ["most_frequent", "stratified"]:
    model = DummyClassifier(strategy=strategy, random_state=0).fit(X_fit, y_fit)
    pred = model.predict(X_valid)
    rows.append({"strategy": strategy, "叫人次數": int(pred.sum()),
                 "accuracy": accuracy_score(y_valid, pred),
                 "precision": precision_score(y_valid, pred, zero_division=0),
                 "recall": recall_score(y_valid, pred)})
print(f"valid {len(valid)} 筆，真的要叫人 {y_valid.sum()} 筆（{y_valid.mean():.3f}）")
print(pd.DataFrame(rows).round(3).to_string(index=False))
