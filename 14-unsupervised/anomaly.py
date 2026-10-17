"""14｜IsolationForest：找出「很少見」的事件交給人看。罕見不等於該通知，也不等於錯。"""
import pandas as pd
from sklearn.ensemble import IsolationForest

from common import NUMERIC, TARGET, fit_valid_split, load_events, time_split

train, _ = time_split(load_events())
fit, valid = fit_valid_split(train)

# 只用前 6 週學「平常長什麼樣子」；不給標籤
iso = IsolationForest(n_estimators=200, random_state=0).fit(fit[NUMERIC])
v = valid.assign(罕見分數=(-iso.score_samples(valid[NUMERIC])).round(3))

print("== 1. 罕見分數的分布（valid，越大越罕見）==")
print(v.罕見分數.describe()[["min", "25%", "50%", "75%", "max"]].round(3).to_string())

print("\n== 2. 最罕見的 8 個 incident（每個 incident 取一筆）==")
cols = ["event_id", "zone", "hour", "confidence", "duration_sec", "light_lux", "prior_violations_30d", "罕見分數"]
top = v.sort_values("罕見分數", ascending=False).drop_duplicates("incident_id").head(8)
print(top[cols].to_string(index=False))

print("\n== 3. 和全體比，它們哪裡不一樣（中位數）==")
compare = pd.DataFrame({"最罕見 1%": v[v.罕見分數 >= v.罕見分數.quantile(0.99)][NUMERIC].median(),
                        "valid 全體": v[NUMERIC].median()})
print(compare.round(1).to_string())

print("\n== 4. 罕見 ≠ 該通知：事後拿標籤對照 ==")
for q in [0.99, 0.95, 0.90]:
    part = v[v.罕見分數 >= v.罕見分數.quantile(q)]
    print(f"最罕見 {1 - q:.0%}（{len(part)} 筆）：該通知比例 {part[TARGET].mean():.3f}，"
          f"其中休息區 {(part.zone == '休息區').sum()} 筆")
print(f"valid 全體（{len(v)} 筆）：該通知比例 {v[TARGET].mean():.3f}")

print("\n== 5. 交給人看的清單：每週最多 10 個 incident ==")
v["週"] = v.event_time.dt.strftime("%G-W%V")
review = (v.sort_values("罕見分數", ascending=False).drop_duplicates("incident_id")
          .groupby("週").head(10).sort_values(["週", "罕見分數"], ascending=[True, False]))
print(review.groupby("週").agg(incident=("incident_id", "size"), 最低罕見分數=("罕見分數", "min")).to_string())
