"""01｜公司現行規則：偵測信心 >= 0.7 就通知現場主管。它錯在哪裡？"""
import pandas as pd

from common import current_rule, load_events

events = load_events()
week = events[(events.event_time >= "2025-12-22") & (events.event_time < "2025-12-29")].copy()
week["rule"] = current_rule(week)

print("== 1. 一週的事件（2025-12-22～12-28）==")
print(f"事件 {len(week)} 筆；規則通知 {week.rule.sum()} 次；陳主任事後認定該通知 {week.needs_notify.sum()} 次")
print(f"平均每天通知 {week.rule.sum() / 7:.1f} 次")

print("\n== 2. 規則和陳主任的判斷對照 ==")
table = pd.crosstab(week.needs_notify.map({1: "該通知", 0: "不必通知"}),
                    week.rule.map({1: "規則通知", 0: "規則不通知"}))
print(table.to_string())

print("\n== 3. 兩種錯，各看 3 筆 ==")
cols = ["event_id", "zone", "shift", "confidence", "duration_sec", "light_lux"]
false_alarm = week[(week.rule == 1) & (week.needs_notify == 0)]
missed = week[(week.rule == 0) & (week.needs_notify == 1)]
print(f"誤報（通知了但不必）：{len(false_alarm)} 筆")
print(false_alarm[cols].head(3).to_string(index=False))
print(f"漏報（該通知卻沒通知）：{len(missed)} 筆")
print(missed[cols].head(3).to_string(index=False))

print("\n== 4. 誤報集中在哪裡 ==")
share = pd.DataFrame({"誤報": false_alarm.zone.value_counts(), "該區全部事件": week.zone.value_counts()})
share["誤報占該區"] = (share["誤報"] / share["該區全部事件"]).round(3)
print(share.sort_values("誤報", ascending=False).to_string())
