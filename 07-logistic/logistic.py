"""07｜邏輯迴歸：輸出機率、讀係數。同一個 Pipeline，只把 KNN 換掉。"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import FEATURES, NUMERIC, TARGET, fit_valid_split, load_events, make_preprocessor, score_row, time_split

train, _ = time_split(load_events())
fit, valid = fit_valid_split(train)
X_fit, y_fit = fit[FEATURES], fit[TARGET]
X_valid, y_valid = valid[FEATURES], valid[TARGET]

model = Pipeline([("prep", make_preprocessor()), ("lr", LogisticRegression())])
model.fit(X_fit, y_fit)

print("== 1. predict_proba：每筆事件一個機率 ==")
proba = model.predict_proba(X_valid)[:, 1]
look = valid[["event_id", "zone", "confidence", "duration_sec", TARGET]].copy()
look["P(該通知)"] = proba.round(3)
look["predict"] = model.predict(X_valid)
print(look.iloc[[0, 7, 56]].to_string(index=False))

print("\n== 2. predict ＝ 機率 >= 0.5 ==")
print("兩者完全相同：", bool((model.predict(X_valid) == (proba >= 0.5)).all()))
bins = pd.cut(proba, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], include_lowest=True)
dist = pd.DataFrame({"事件數": pd.Series(bins).value_counts(sort=False).values,
                     "實際該通知比例": pd.Series(y_valid.values).groupby(bins, observed=False).mean().round(3).values},
                    index=[str(b) for b in bins.categories])
print(dist.to_string())

print("\n== 3. 換一個門檻，通知次數就變了 ==")
for t in [0.3, 0.5, 0.7]:
    r = score_row(f"門檻 {t}", y_valid, (proba >= t).astype(int))
    print(f"門檻 {t}：通知 {r['alerts']} 次，precision {r['precision']:.3f}，recall {r['recall']:.3f}")

print("\n== 4. 係數：正的推向「通知」，負的推向「不通知」==")
coef = pd.Series(model["lr"].coef_[0], index=model["prep"].get_feature_names_out())
coef.index = coef.index.str.replace("num__", "").str.replace("cat__", "")
print(coef.sort_values().round(3).to_string())
print(f"截距 {model['lr'].intercept_[0]:.3f}")

print("\n== 5. E02916 的分數是怎麼加出來的 ==")
x = model["prep"].transform(valid[valid.event_id == "E02916"][FEATURES])
x = np.asarray(x.todense() if hasattr(x, "todense") else x)[0]
part = pd.Series(x * model["lr"].coef_[0], index=coef.index)
part = part[part.abs() >= 0.05].sort_values(key=abs, ascending=False)
print(part.round(3).to_string())
total = model["lr"].intercept_[0] + (x * model["lr"].coef_[0]).sum()
print(f"其他欄位合計 {total - model['lr'].intercept_[0] - part.sum():+.3f}；截距 {model['lr'].intercept_[0]:+.3f}")
print(f"總分 {total:.3f} → 機率 1/(1+e^-總分) = {1 / (1 + np.exp(-total)):.3f}")

print("\n== 6. 動一個欄位，看機率怎麼變（E02916）==")
base = valid[valid.event_id == "E02916"][FEATURES]
sd = fit["duration_sec"].std()
tries = {"原樣": base,
         f"持續時間 +{sd:.0f} 秒": base.assign(duration_sec=base.duration_sec + sd),
         "改在休息區": base.assign(zone="休息區"),
         "照度 -235 lux": base.assign(light_lux=base.light_lux - 235)}
for name, x in tries.items():
    print(f"{name:<12} P(該通知) = {model.predict_proba(x)[0, 1]:.3f}")

print("\n== 7. 沒縮放的係數不能比大小 ==")
raw = LogisticRegression(max_iter=5000).fit(fit[NUMERIC], y_fit)
scaled = coef[NUMERIC]
print(pd.DataFrame({"未縮放": raw.coef_[0].round(4), "已標準化": scaled.round(3).values}, index=NUMERIC).to_string())

print("\n== 8. 成績表（valid）==")
print(pd.DataFrame([score_row("邏輯迴歸", y_valid, model.predict(X_valid))]).to_string(index=False))
