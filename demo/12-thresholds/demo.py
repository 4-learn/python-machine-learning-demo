"""12｜scikit-learn 的 precision_recall_curve：0.8 和 0.5 這兩個門檻從哪裡來。"""
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

print("== 1. predict 只是 predict_proba >= 0.5 ==")
train = pd.DataFrame({"duration_seconds": [3, 5, 8, 12, 15, 18, 22, 25, 30, 40, 55, 70],
                      "label":            [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]})
model = LogisticRegression().fit(train[["duration_seconds"]], train["label"])
new = pd.DataFrame({"duration_seconds": [6, 20, 28, 60]})
proba = model.predict_proba(new)[:, 1]
print(new.assign(score=proba.round(3), predict=model.predict(new), score_ge_05=(proba >= 0.5).astype(int))
      .to_string(index=False))

# valid 上 21 筆告警：score 是模型給的分數，label 是事後的結果（10 筆真的）
valid = pd.DataFrame({
    "score": [0.97, 0.93, 0.91, 0.88, 0.84, 0.81, 0.76, 0.70, 0.66, 0.62, 0.58,
              0.52, 0.47, 0.41, 0.35, 0.30, 0.24, 0.18, 0.12, 0.08, 0.05],
    "label": [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0,
              0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
})
y, score = valid["label"], valid["score"]

print("\n== 2. 手動掃幾個門檻 ==")
rows = []
for t in [0.9, 0.8, 0.5, 0.3]:
    called = score >= t
    tp = int((called & (y == 1)).sum())
    rows.append({"門檻": t, "叫人": int(called.sum()), "其中真的": tp,
                 "precision": round(tp / called.sum(), 3), "recall": round(tp / y.sum(), 3)})
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))

print("\n== 3. precision_recall_curve：一次掃完所有門檻 ==")
precision, recall, thresholds = precision_recall_curve(y, score)
print(f"precision {len(precision)} 個、recall {len(recall)} 個、thresholds {len(thresholds)} 個")
curve = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1]})
print(f"最後一組：precision {precision[-1]:.3f}、recall {recall[-1]:.3f}（沒有對應的門檻）")

print("\n== 4. 去掉最後一個，三個陣列對齊成一張表 ==")
print(curve.to_string(index=False, float_format="%.3f"))

print("\n== 5. 選 HIGH：precision >= 0.9 的門檻裡，最低的那個 ==")
ok = curve[curve["precision"] >= 0.9]
high = float(ok["threshold"].min())
print(f"符合的門檻 {len(ok)} 個，HIGH = {high}")

print("\n== 6. 選 LOW：recall >= 0.9（漏掉不超過 10%）的門檻裡，最高的那個 ==")
ok = curve[curve["recall"] >= 0.9]
low = float(ok["threshold"].max())
missed = int(((score < low) & (y == 1)).sum())
print(f"符合的門檻 {len(ok)} 個，LOW = {low}；低於 LOW 的真事件 {missed} 筆／共 {y.sum()} 筆")

print("\n== 7. 用兩個門檻切三段，寫成設定 ==")
valid["decision"] = pd.cut(score, bins=[-0.01, low, high, 1.01], right=False,
                           labels=["log_only", "require_review", "auto_alert"])
print(valid.groupby("decision", observed=False)["label"].agg(告警數="count", 其中真的="sum").to_string())
config = {"auto_alert_min_precision": 0.9, "max_missed_rate": 0.1, "high": high, "low": low}
print(json.dumps(config, ensure_ascii=False))
