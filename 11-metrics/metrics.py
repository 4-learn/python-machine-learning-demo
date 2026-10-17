"""11｜混淆矩陣到 precision／recall：把「答對幾題」拆成「錯在哪裡、往哪個方向錯」。"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from common import FEATURES, TARGET, current_rule, fit_valid_split, load_events, make_preprocessor, time_split

train, _ = time_split(load_events())
fit, valid = fit_valid_split(train)
y_valid = valid[TARGET]

# 10 節調參後的梯度提升（門檻還是預設的 0.5）
model = make_pipeline(make_preprocessor(scale=False),
                      HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_leaf_nodes=15,
                                                     random_state=0))
model.fit(fit[FEATURES], fit[TARGET])
preds = {"多數類": pd.Series(0, index=valid.index),
         "現行規則": current_rule(valid),
         "梯度提升": pd.Series(model.predict(valid[FEATURES]), index=valid.index)}


def error_map(y_true, y_pred):
    table = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0, 1]),
                         index=["實際：不必通知", "實際：該通知"], columns=["預測：不通知", "預測：通知"])
    return table.to_string()


print("== 1. sklearn 的混淆矩陣（梯度提升，valid 963 筆）==")
cm = confusion_matrix(y_valid, preds["梯度提升"])
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"tn={tn} fp={fp} fn={fn} tp={tp}")

print("\n== 2. 同一張表，加上名字 ==")
print(error_map(y_valid, preds["梯度提升"]))

print("\n== 3. 三張錯誤地圖 ==")
for name, p in preds.items():
    print(f"[{name}]  accuracy {(p == y_valid).mean():.3f}")
    print(error_map(y_valid, p))

print("\n== 4. 從四個格子算 precision、recall ==")
for name, p in preds.items():
    tn, fp, fn, tp = confusion_matrix(y_valid, p, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    print(f"{name}：通知 {tp + fp} 次、其中真的 {tp} 次 → precision {precision:.3f}；"
          f"該通知 {tp + fn} 次、抓到 {tp} 次 → recall {recall:.3f}")

print("\n== 5. 錯誤的方向：漏報與誤報各在哪一區（梯度提升）==")
v = valid.assign(pred=preds["梯度提升"])
by_zone = pd.DataFrame({
    "該通知": v.groupby("zone")[TARGET].sum(),
    "漏報": v[(v[TARGET] == 1) & (v.pred == 0)].zone.value_counts(),
    "誤報": v[(v[TARGET] == 0) & (v.pred == 1)].zone.value_counts(),
}).fillna(0).astype(int)
print(by_zone.sort_values("該通知", ascending=False).to_string())

print("\n== 6. classification_report ==")
print(classification_report(y_valid, preds["梯度提升"], target_names=["不必通知", "該通知"], digits=3))
