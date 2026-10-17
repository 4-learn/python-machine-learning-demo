"""12｜門檻怎麼選：用有標籤的驗證集＋業務代價，回答「0.7 從哪裡來？」"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.pipeline import make_pipeline

from common import FEATURES, TARGET, current_rule, fit_valid_split, load_events, make_preprocessor, time_split

FN_COST, FP_COST = 10, 1          # 陳主任：漏報一次的代價＝誤報 10 次
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

train, _ = time_split(load_events())
fit, valid = fit_valid_split(train)
y = valid[TARGET].to_numpy()

model = make_pipeline(make_preprocessor(scale=False),
                      HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_leaf_nodes=15,
                                                     random_state=0))
model.fit(fit[FEATURES], fit[TARGET])
proba = model.predict_proba(valid[FEATURES])[:, 1]


def row(score, t, fn_cost=FN_COST, fp_cost=FP_COST):
    tn, fp, fn, tp = confusion_matrix(y, (score >= t).astype(int), labels=[0, 1]).ravel()
    return {"門檻": t, "通知": int(tp + fp), "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3), "漏報": int(fn), "誤報": int(fp),
            "成本": int(fn_cost * fn + fp_cost * fp)}


print("== 1. 現行規則（confidence >= 0.7）的帳 ==")
r = row(valid.confidence.to_numpy(), 0.7)
print(f"通知 {r['通知']} 次：漏報 {r['漏報']}、誤報 {r['誤報']} → 成本 {FN_COST}×{r['漏報']} + {r['誤報']} = {r['成本']}")
print(f"同一條規則，0.7 換成 0.6：成本 {row(valid.confidence.to_numpy(), 0.6)['成本']}")

print("\n== 2. PR 曲線：門檻由高往低，precision 與 recall 的交換 ==")
prec, rec, thr = precision_recall_curve(y, proba)
pts = []
for target in [0.5, 0.7, 0.8, 0.9, 0.95]:
    i = np.where(rec[:-1] >= target)[0][-1]      # 達到這個 recall 的最高門檻
    pts.append({"想抓到的 recall": target, "門檻最多可以設到": round(float(thr[i]), 3),
                "precision": round(float(prec[i]), 3)})
print(pd.DataFrame(pts).to_string(index=False))
print(f"average precision（PR 曲線下面積）：模型 {average_precision_score(y, proba):.3f}；"
      f"只用 confidence {average_precision_score(y, valid.confidence):.3f}")

print("\n== 3. 在 valid 上掃門檻：成本 = 10×漏報 + 1×誤報 ==")
rows = [row(proba, t) for t in GRID]
print(pd.DataFrame(rows).to_string(index=False))
best = min(rows, key=lambda r: r["成本"])
cost_at = {r["門檻"]: r["成本"] for r in rows}
print(f"最低成本：門檻 {best['門檻']:.2f}，成本 {best['成本']}（0.5 是 {cost_at[0.5]}；0.7 是 {cost_at[0.7]}）")

print("\n== 4. 代價比例換了，門檻跟著換 ==")
out = []
for ratio in [1, 3, 10, 30]:
    b = min((row(proba, g, fn_cost=ratio) for g in GRID), key=lambda r: r["成本"])
    out.append({"漏報:誤報": f"{ratio}:1", "最佳門檻": b["門檻"], "通知": b["通知"], "漏報": b["漏報"], "誤報": b["誤報"]})
print(pd.DataFrame(out).to_string(index=False))

print("\n== 5. 把決定寫下來（設定檔範例，只印出來，不覆寫 config/）==")
print(f"""threshold:
  value: {best['門檻']:.2f}
  chosen_on: 驗證集 {valid.event_time.min():%Y-%m-%d}～{valid.event_time.max():%Y-%m-%d}（{len(valid)} 筆）
  model_trained_on: {fit.event_time.min():%Y-%m-%d}～{fit.event_time.max():%Y-%m-%d}（{len(fit)} 筆）
  rule: 最小化 {FN_COST}×漏報 + {FP_COST}×誤報（驗證集成本 {best['成本']}）
  approved_by: 待陳主任核准""")
