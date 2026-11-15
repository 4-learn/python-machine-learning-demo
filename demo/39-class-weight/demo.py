"""39｜scikit-learn 的 class_weight：少數類（要叫人的）只占一成多時，怎麼讓模型多在意它。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

alerts = pd.read_csv("story/data/factory/alerts.csv")
COLS = ["confidence", "duration_seconds", "prior_alerts_30d"]
fit = alerts[alerts["occurred_at"] < "2025-12-15"]
valid = alerts[(alerts["occurred_at"] >= "2025-12-15") & (alerts["occurred_at"] < "2025-12-29")]


def row(name, pred):
    return {"方法": name, "precision": round(precision_score(valid["label"], pred, zero_division=0), 3),
            "recall": round(recall_score(valid["label"], pred), 3), "叫人": int(pred.sum())}


print("== 1. 少數類：要叫人的只占一成多 ==")
print(f"fit {len(fit)} 筆：label=1 有 {fit['label'].sum()} 筆（{fit['label'].mean():.3f}）")

print("\n== 2. compute_class_weight('balanced')：權重怎麼算 ==")
w = compute_class_weight("balanced", classes=np.array([0, 1]), y=fit["label"])
print(f"0 → {w[0]:.3f}、1 → {w[1]:.3f}")
n, n0, n1 = len(fit), (fit["label"] == 0).sum(), (fit["label"] == 1).sum()
print(f"手算：{n} / (2 × {n0}) = {n / (2 * n0):.3f}；{n} / (2 × {n1}) = {n / (2 * n1):.3f}")

print("\n== 3. class_weight：None、balanced、dict ==")
rows = []
for cw in [None, "balanced", {0: 1, 1: 10}]:
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight=cw)).fit(fit[COLS], fit["label"])
    rows.append(row(f"class_weight={cw}", model.predict(valid[COLS])))
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 4. 另一條路：不加權，只把門檻往下調（12 節）==")
plain = make_pipeline(StandardScaler(), LogisticRegression()).fit(fit[COLS], fit["label"])
score = plain.predict_proba(valid[COLS])[:, 1]
rows = [row(f"門檻 {t}", (score >= t).astype(int)) for t in [0.5, 0.3, 0.2, 0.1]]
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 5. 加權改的是分數，排名幾乎不變 ==")
bal = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(fit[COLS], fit["label"])
score_bal = bal.predict_proba(valid[COLS])[:, 1]
print(f"平均分數：不加權 {score.mean():.3f}、balanced {score_bal.mean():.3f}")
same = pd.Series(score).rank().corr(pd.Series(score_bal).rank())
print(f"兩組分數的排名相關：{same:.3f}")
