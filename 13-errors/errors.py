"""13｜錯誤分析：用 12 節的門檻 0.20，把漏報與誤報拉出來分組，再看 permutation importance。"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline

from common import FEATURES, TARGET, fit_valid_split, load_events, make_preprocessor, time_split

THRESHOLD = 0.20   # 12 節在 valid 上選的門檻（config/notify.yaml）

train, _ = time_split(load_events())
fit, valid = fit_valid_split(train)
model = make_pipeline(make_preprocessor(scale=False),
                      HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_leaf_nodes=15,
                                                     random_state=0))
model.fit(fit[FEATURES], fit[TARGET])

v = valid.copy()
v["proba"] = model.predict_proba(v[FEATURES])[:, 1].round(3)
v["pred"] = (v.proba >= THRESHOLD).astype(int)
v["結果"] = np.select([(v.pred == 1) & (v[TARGET] == 1), (v.pred == 0) & (v[TARGET] == 1),
                       (v.pred == 1) & (v[TARGET] == 0)], ["抓到", "漏報", "誤報"], "沒事")

print(f"== 1. 門檻 {THRESHOLD} 下的四種結果（valid {len(v)} 筆）==")
summary = v.groupby("結果").agg(事件=("event_id", "size"), incident=("incident_id", "nunique"))
print(summary.loc[["抓到", "漏報", "誤報", "沒事"]].to_string())

print("\n== 2. 四種結果的「典型樣子」（中位數）==")
cols = ["confidence", "duration_sec", "light_lux", "prior_violations_30d", "proba"]
print(v.groupby("結果")[cols].median().loc[["抓到", "漏報", "誤報", "沒事"]].round(3).to_string())

print("\n== 3. 依持續時間切開 ==")
v["持續"] = pd.cut(v.duration_sec, [0, 25, 40, 60, 1000], labels=["<25 秒", "25–40", "40–60", ">60"])
print(pd.crosstab(v.持續, v.結果)[["抓到", "漏報", "誤報", "沒事"]].to_string())

print("\n== 4. 依區域切開 ==")
print(pd.crosstab(v.zone, v.結果)[["抓到", "漏報", "誤報"]].to_string())

print("\n== 5. 11 筆漏報逐筆看 ==")
show = ["event_id", "incident_id", "zone", "confidence", "duration_sec", "prior_violations_30d", "proba"]
print(v[v.結果 == "漏報"][show].to_string(index=False))

print("\n== 6. 誤報機率最高的 5 筆 ==")
print(v[v.結果 == "誤報"].sort_values("proba", ascending=False)[show].head(5).to_string(index=False))

print("\n== 7. permutation importance（valid，打亂一欄後 average precision 掉多少）==")
r = permutation_importance(model, valid[FEATURES], valid[TARGET], scoring="average_precision",
                           n_repeats=10, random_state=0)
imp = pd.DataFrame({"平均下降": r.importances_mean, "標準差": r.importances_std}, index=FEATURES)
print(imp.sort_values("平均下降", ascending=False).round(3).to_string())
