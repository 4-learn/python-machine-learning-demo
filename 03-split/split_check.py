"""03｜切完先檢查：形狀、類別比例；隨機切割 vs common.time_split。"""
import pandas as pd
from sklearn.model_selection import train_test_split

from common import TARGET, load_events, time_split

events = load_events()

print("== 1. 隨機切割 75%／25%，先看形狀 ==")
train, test = train_test_split(events, test_size=0.25, random_state=0)
print(f"全部 {events.shape}；train {train.shape}；test {test.shape}")

print("\n== 2. 再看類別比例：換 5 個 random_state ==")
rows = []
for seed in range(5):
    _, plain = train_test_split(events, test_size=0.25, random_state=seed)
    _, strat = train_test_split(events, test_size=0.25, random_state=seed, stratify=events[TARGET])
    rows.append({"random_state": seed, "不分層 test 正例": round(plain[TARGET].mean(), 3),
                 "stratify test 正例": round(strat[TARGET].mean(), 3)})
print(pd.DataFrame(rows).to_string(index=False))
print(f"全部資料正例比例 {events[TARGET].mean():.3f}")

print("\n== 3. 全課的切割：common.time_split ==")
train, test = time_split(events)
for name, part in [("train", train), ("test", test)]:
    print(f"{name}: {len(part)} 筆，{part.event_time.min():%Y-%m-%d}～{part.event_time.max():%Y-%m-%d}，"
          f"正例 {part[TARGET].mean():.3f}")
shared = set(train.incident_id) & set(test.incident_id)
print(f"同時出現在兩邊的 incident：{len(shared)} 起")
print(f"同時出現在兩邊的人員：{len(set(train.worker_id) & set(test.worker_id))} 人（共 {events.worker_id.nunique()} 人）")
