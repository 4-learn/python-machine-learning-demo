"""13｜是模型錯，還是標籤錯？用標註規則與「交叉驗證的預測」找可疑標籤。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from common import FEATURES, TARGET, fit_valid_split, load_events, make_preprocessor, time_split

THRESHOLD = 0.20


def make_model():
    return make_pipeline(make_preprocessor(scale=False),
                         HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_leaf_nodes=15,
                                                        random_state=0))


def breaks_rule(df):
    """陳主任的標註規則：休息區不通知；持續 25 秒以上才算。量測有誤差，所以用 20 秒當「明顯太短」。"""
    return (df[TARGET] == 1) & ((df.zone == "休息區") | (df.duration_sec < 20))


events = load_events()
train, _ = time_split(events)
fit, valid = fit_valid_split(train)

print("== 1. 和標註規則矛盾的「該通知」（全部 10 週）==")
bad = events[breaks_rule(events)]
pos = events[events[TARGET] == 1]
print(f"標成該通知 {len(pos)} 筆，其中違反規則 {len(bad)} 筆（{bad.incident_id.nunique()} 個 incident）")
print(f"  休息區 {(bad.zone == '休息區').sum()} 筆；不在休息區但持續不到 20 秒 {(bad.zone != '休息區').sum()} 筆")

print("\n== 2. 模型很有把握、卻和標籤相反的事件（train，5 折分組交叉驗證）==")
oof = cross_val_predict(make_model(), train[FEATURES], train[TARGET], groups=train.incident_id,
                        cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
t = train.assign(proba=oof.round(3))
low = t[(t[TARGET] == 1) & (t.proba < 0.05)]
high = t[(t[TARGET] == 0) & (t.proba > 0.90)]
print(f"標「該通知」但機率 < 0.05：{len(low)} 筆；其中也違反標註規則 {breaks_rule(low).sum()} 筆")
print(f"標「不必通知」但機率 > 0.90：{len(high)} 筆")
show = ["event_id", "zone", "confidence", "duration_sec", "prior_violations_30d", TARGET, "proba"]
print(high[show].to_string(index=False))

print("\n== 3. 回頭看 valid 的 11 筆漏報 ==")
model = make_model().fit(fit[FEATURES], fit[TARGET])
v = valid.assign(proba=model.predict_proba(valid[FEATURES])[:, 1])
missed = v[(v.proba < THRESHOLD) & (v[TARGET] == 1)]
print(f"漏報 {len(missed)} 筆，其中違反標註規則 {breaks_rule(missed).sum()} 筆")


def tally(y_true):
    pred = v.proba >= THRESHOLD
    fn, fp = int((~pred & (y_true == 1)).sum()), int((pred & (y_true == 0)).sum())
    return {"漏報": fn, "誤報": fp, "成本": 10 * fn + fp}


fixed = v[TARGET].where(~breaks_rule(v), 0)
print(pd.DataFrame([{"valid 標籤": "原始", **tally(v[TARGET])},
                    {"valid 標籤": "違反規則者改成 0", **tally(fixed)}]).to_string(index=False))

print("\n== 4. 把 fit 裡違反規則的標籤修掉再訓練 ==")
rows = []
for name, part in [("原始 fit", fit),
                   ("修正 fit 標籤", fit.assign(**{TARGET: fit[TARGET].where(~breaks_rule(fit), 0)}))]:
    p = make_model().fit(part[FEATURES], part[TARGET]).predict_proba(valid[FEATURES])[:, 1]
    pred = p >= THRESHOLD
    cost = {}
    for label, y in [("原始", valid[TARGET]), ("修正", fixed)]:
        y = y.to_numpy()
        cost[f"valid {label}標籤成本"] = int(10 * (~pred & (y == 1)).sum() + (pred & (y == 0)).sum())
    rows.append({"訓練資料": name, "改了幾筆": int((part[TARGET] != fit[TARGET]).sum()), **cost})
print(pd.DataFrame(rows).to_string(index=False))
