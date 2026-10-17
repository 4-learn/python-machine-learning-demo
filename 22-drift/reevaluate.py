"""22｜偵測到漂移之後：重新標註一批 → 重新評估 → 寫提案，等人核准。

    python 22-drift/reevaluate.py

relabel-C.csv：C 工地 01-26 之後全部 147 筆，由陳主任依同一份標註規則補標（2026-02-10）。
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "21-pipeline"))
from pipeline import CONFIG, NotifyPipeline, read_yaml  # noqa: E402

from common import DATA, load_events, score_row, time_split  # noqa: E402

pipe = NotifyPipeline()
new = load_events(DATA / "events-new-weeks.csv")
after = new.event_time >= pd.Timestamp("2026-01-26", tz="Asia/Taipei")
c_after = new[(new.site == "C") & after]


def cost(y, proba, t):
    alert = proba >= t
    return int(10 * ((~alert) & (y == 1)).sum() + (alert & (y == 0)).sum())


print("== 1. 只用原本就有的標註（約 25%）==")
known = c_after.dropna(subset=["needs_notify"])
p = pipe.proba(known)
print(f"C 工地 01-26 之後 {len(c_after)} 筆，已標註 {len(known)} 筆，其中需要通知 {int(known.needs_notify.sum())} 筆")
print(score_row("模型 @0.2", known.needs_notify.astype(int), (p >= pipe.threshold).astype(int)))

print("\n== 2. 重新標註全部 147 筆之後 ==")
relabel = pd.read_csv(HERE / "relabel-C.csv")
c = c_after.drop(columns="needs_notify").merge(relabel[["event_id", "needs_notify"]], on="event_id")
assert len(c) == len(c_after)
same = known[["event_id", "needs_notify"]].merge(relabel, on="event_id", suffixes=("_舊", "_新"))
print(f"原本已標註的 {len(same)} 筆，新舊標註一致：{(same.needs_notify_舊 == same.needs_notify_新).sum()} 筆")
p = pipe.proba(c)
print(score_row("模型 @0.2", c.needs_notify, (p >= pipe.threshold).astype(int)))
print(score_row("規則 conf≥0.7", c.needs_notify, (c.confidence >= 0.7).astype(int)))

print("\n== 3. 對照：測試期（12-29～01-11）C 工地，攝影機還沒換 ==")
_, test = time_split(load_events())
ct = test[test.site == "C"]
pt = pipe.proba(ct)
print(score_row("模型 @0.2", ct.needs_notify, (pt >= pipe.threshold).astype(int)), "成本", cost(ct.needs_notify, pt, 0.2))

print("\n== 4. 漏掉的是哪些 ==")
miss = c[(p < pipe.threshold) & (c.needs_notify == 1)].assign(proba=p.round(3))
print(miss[["event_id", "zone", "confidence", "duration_sec", "prior_violations_30d", "proba"]].to_string(index=False))

print("\n== 5. 在重新標註的資料上，換幾個門檻看看（成本 = 10×漏報 + 1×誤報）==")
rows = []
for t in [0.05, 0.1, 0.15, 0.2, 0.3]:
    s = score_row(f"@{t}", c.needs_notify, (p >= t).astype(int))
    rows.append({"門檻": t, "通知": s["alerts"], "precision": s["precision"], "recall": s["recall"], "成本": cost(c.needs_notify, p, t)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 6. 寫提案（不是改設定）==")
proposal = read_yaml(CONFIG)
proposal["status"] = "proposed"
proposal["threshold"] = {
    "value": 0.2,
    "site_override": {"C": 0.1},
    "chosen_on": "relabel-C-2026-02（C 工地 2026-01-26～02-08，147 筆全數重新標註）",
    "rule": "最小化 10×漏報 + 1×誤報（C 工地成本 61 → 28）",
    "reason": "C 工地 2026-01-26 換攝影機，confidence 整體下降（PSI 第 3 週 1.210、第 4 週 1.416）",
    "approved_by": None,
    "approved_at": None,
}
out = HERE / "notify-proposed.yaml"
out.write_text(yaml.safe_dump(proposal, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"已寫 {out.name}：status={proposal['status']}，approved_by={proposal['threshold']['approved_by']}")
print(f"config/notify.yaml 沒有動：threshold.value={read_yaml(CONFIG)['threshold']['value']}，"
      f"approved_by={read_yaml(CONFIG)['threshold']['approved_by']}")
