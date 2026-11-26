"""04｜scikit-learn DummyClassifier：不用模型能拿幾分。順便認識 fit／predict 與三個分數。"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 用 10 筆告警示範 DummyClassifier
print("== 1. 10 筆告警：fit 學、predict 答 ==")
X = pd.DataFrame({"confidence": [0.95, 0.83, 0.81, 0.72, 0.66, 0.55, 0.49, 0.41, 0.35, 0.92]})
y = pd.Series([1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X, y)
print("predict：", dummy.predict(X).tolist())

# 用同樣 10 筆告警，算三個分數
print("\n== 2. 三個分數 ==")
# 平台規則：信心值 0.8 以上就叫
rule = (X["confidence"] >= 0.8).astype(int)

# 計算三個分數
names = ["多數類", "conf>=0.8"]
preds = [dummy.predict(X), rule]

for i in range(2):
    name = names[i]
    pred = preds[i]
    print(name,
          "accuracy", round(accuracy_score(y, pred), 3),
          "precision", round(precision_score(y, pred, zero_division=0), 3),
          "recall", round(recall_score(y, pred), 3))

# 用工廠線的告警示範 DummyClassifier
print("\n== 3. 換成工廠線：前 6 週 fit、第 7–8 週 predict ==")
df = pd.read_csv("story/data/factory/alerts.csv")
df["occurred_at"] = pd.to_datetime(df["occurred_at"])
fit = df[df["occurred_at"] < "2025-12-15"]
valid = df[(df["occurred_at"] >= "2025-12-15") & (df["occurred_at"] < "2025-12-29")]

# 用 DummyClassifier 預測 valid
X_fit, y_fit = fit[["confidence"]], fit["label"]
X_valid, y_valid = valid[["confidence"]], valid["label"]
model = DummyClassifier(strategy="most_frequent").fit(X_fit, y_fit)

pred = model.predict(X_valid)
print("valid", len(valid), "筆，真的要叫人", y_valid.sum(), "筆，比例", round(y_valid.mean(), 3))
print("多數類：叫人", pred.sum(), "次",
      "accuracy", round(accuracy_score(y_valid, pred), 3),
      "precision", round(precision_score(y_valid, pred, zero_division=0), 3),
      "recall", round(recall_score(y_valid, pred), 3))
