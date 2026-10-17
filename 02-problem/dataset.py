"""02｜把「要不要通知」寫成 X 與 y：一列是什麼、y 從哪裡來、哪些欄位是答案的影子。"""
import pandas as pd

from common import CATEGORICAL, FEATURES, LEAKY, NUMERIC, TARGET, load_events

events = load_events()

print("== 1. 一列是什麼 ==")
print(f"{events.shape[0]} 列、{events.shape[1]} 欄")
print(events.iloc[1].to_string())

print("\n== 2. 標籤 y：陳主任事後認定要不要通知 ==")
counts = events[TARGET].value_counts().sort_index()
print(f"不必通知 {counts[0]} 筆、該通知 {counts[1]} 筆，正例比例 {events[TARGET].mean():.3f}")

print("\n== 3. 同一起 incident 會有好幾列 ==")
size = events.groupby("incident_id").size()
print(f"incident {len(size)} 起；每起的列數：", size.value_counts().sort_index().to_dict())
example = size[size == 3].index[0]
cols = ["event_id", "incident_id", "event_time", "confidence", "duration_sec", "needs_notify"]
print(events.loc[events.incident_id == example, cols].to_string(index=False))

print("\n== 4. 答案的影子：被通知之後才有的欄位 ==")
has_ticket = events.ticket_id.notna().map({True: "有工單", False: "沒有工單"})
print(pd.crosstab(has_ticket, events[TARGET].map({1: "該通知", 0: "不必通知"})).to_string())
notified = pd.to_datetime(events.notified_at, format="%Y-%m-%d %H:%M:%S%z")
lag = (notified - events.event_time).dt.total_seconds() / 60
print(f"notified_at 比事件晚 {lag.min():.0f}～{lag.max():.0f} 分鐘；handled_minutes 有值 {events.handled_minutes.notna().sum()} 筆")

print("\n== 5. 每一欄的角色 ==")
role = {c: "識別碼" for c in ["event_id", "incident_id", "worker_id"]}
role["event_time"] = "時間（拿來切割，hour 已從它算出）"
role.update({c: "特徵：數值" for c in NUMERIC})
role.update({c: "特徵：類別" for c in CATEGORICAL})
role.update({c: "洩漏：事後才有" for c in LEAKY})
role[TARGET] = "標籤 y"
table = pd.Series(role, name="角色").reindex(events.columns)
print(table.to_string())
chosen = [c for c in events.columns if table[c].startswith("特徵")]
print(f"特徵 {len(chosen)} 欄，與 common.FEATURES 相同：{sorted(chosen) == sorted(FEATURES)}")

print("\n== 6. 類別欄：同一欄的不同值，正例比例差很多 ==")
by_zone = events.groupby("zone")[TARGET].agg(["size", "mean"]).round(3)
by_zone.columns = ["事件數", "該通知比例"]
print(by_zone.sort_values("該通知比例").to_string())

print("\n== 7. 如果先 dropna() 再說 ==")
dropped = events.dropna()
print(f"dropna 之後剩 {len(dropped)} 列；needs_notify 的值：", dropped[TARGET].value_counts().to_dict())
