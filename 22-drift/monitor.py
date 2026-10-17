"""22｜上線 4 週的監控：告警率、confidence 分布、PSI、變化點。只看資料，不改任何設定。

    python 22-drift/monitor.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "21-pipeline"))
from drift import change_point, decile_bins, normal_swing, psi, quantiles  # noqa: E402
from pipeline import NotifyPipeline  # noqa: E402

from common import DATA, NUMERIC, load_events, time_split  # noqa: E402

train, _ = time_split(load_events())
new = load_events(DATA / "events-new-weeks.csv")
new["週"] = (new.event_time - pd.Timestamp("2026-01-12", tz="Asia/Taipei")).dt.days // 7 + 1
train["週"] = (train.event_time - pd.Timestamp("2025-11-03", tz="Asia/Taipei")).dt.days // 7 + 1
scored = NotifyPipeline().score_table(new, cooldown=False)  # 21 節同一條管線（離線路徑）

print("== 1. 告警率：每週、每個工地（模型建議通知的比例）==")
rate = scored.pivot_table(index="週", columns="site", values="model_notify", aggfunc="mean").round(3)
rate["全部"] = scored.groupby("週").model_notify.mean().round(3)
print(rate.to_string())
base = NotifyPipeline().score_table(train, cooldown=False)
print("訓練期（前 8 週）：", base.groupby("site").model_notify.mean().round(3).to_dict(), f"全部 {base.model_notify.mean():.3f}")

print("\n== 2. C 工地 confidence 的分位數：訓練期 vs 上線後每週 ==")
c_train = train[train.site == "C"].confidence
rows = {"訓練期": quantiles(c_train)}
for w in [1, 2, 3, 4]:
    rows[f"第 {w} 週"] = quantiles(new[(new.site == "C") & (new.週 == w)].confidence)
table = pd.DataFrame(rows).T.round(3)
table["筆數"] = [len(c_train)] + [int(((new.site == "C") & (new.週 == w)).sum()) for w in [1, 2, 3, 4]]
print(table.to_string())

print("\n== 3. PSI：每個工地、每週，與訓練期比 ==")
psi_new = pd.DataFrame({s: [round(psi(train[train.site == s].confidence, new[(new.site == s) & (new.週 == w)].confidence), 3)
                            for w in [1, 2, 3, 4]] for s in "ABCD"}, index=[1, 2, 3, 4]).rename_axis("週")
print(psi_new.to_string())

print("\n== 4. 什麼叫「正常的晃動」：訓練期 8 週，每週和整段比的 PSI ==")
swing = {}
for s in "ABCD":
    t = train[train.site == s]
    weekly = [round(psi(t.confidence, t[t.週 == w].confidence), 3) for w in range(1, 9)]
    swing[s] = normal_swing(t)
    print(f"{s} 工地：{weekly}，最大 {swing[s]:.3f}")
level = pd.DataFrame({s: ["超出以往" if v > swing[s] else "-" for v in psi_new[s]] for s in "ABCD"}, index=psi_new.index)
print("上線後每週 PSI 超過該工地以往最大晃動的：")
print(level.to_string())

print("\n== 5. 找變化點：哪一天前後的 confidence 分布差最多（前後各至少 7 天）==")
for s in "ABCD":
    day, gap, n_before, n_after = change_point(new[new.site == s], min_days=7)
    print(f"{s} 工地：{day}，差距 {gap:.3f}（之前 {n_before} 筆、之後 {n_after} 筆）")

print("\n== 6. C 工地每天的最高 confidence ==")
c = new[new.site == "C"]
daily = c.groupby(c.event_time.dt.strftime("%m-%d")).confidence.max().round(3)
print("01-19～01-25：", daily.loc["01-19":"01-25"].tolist())
print("01-26～02-01：", daily.loc["01-26":"02-01"].tolist())
print(f"01-26 之前最高 {c[c.event_time < '2026-01-26'].confidence.max():.3f}；之後最高 {c[c.event_time >= '2026-01-26'].confidence.max():.3f}")

print("\n== 7. 其他欄位也看一下：全部工地，每週 PSI（訓練期十分位數分箱）==")
cols = {f: [round(psi(train[f], new[new.週 == w][f], decile_bins(train[f])), 3) for w in [1, 2, 3, 4]] for f in NUMERIC}
print(pd.DataFrame(cols, index=[1, 2, 3, 4]).rename_axis("週").T.to_string())
print("prior_violations_30d 每週平均：", new.groupby("週").prior_violations_30d.mean().round(2).tolist(),
      f"；訓練期 {train.prior_violations_30d.mean():.2f}")
