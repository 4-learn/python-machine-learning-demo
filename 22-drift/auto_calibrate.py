"""22｜反例：舊版「每 N 筆自動用 KMeans 校準門檻」重跑在上線 4 週的資料上。

    python 22-drift/auto_calibrate.py

舊教材的做法：累積事件 → KMeans(3) 分 confidence → 取中、高兩群中心的中點當新門檻 → 立刻生效。
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_events, score_row  # noqa: E402

EVERY = 100  # 每 100 筆校準一次

new = load_events(DATA / "events-new-weeks.csv").sort_values("event_time", kind="stable").reset_index(drop=True)


def kmeans_threshold(confidence: pd.Series) -> float:
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(confidence.to_frame())
    low, mid, high = sorted(km.cluster_centers_.ravel())
    return round(float((mid + high) / 2), 3)


thresholds, current = [], 0.7  # 一開始用人工設定的 0.7
for i in range(len(new)):
    if i > 0 and i % EVERY == 0:
        current = kmeans_threshold(new.confidence.iloc[:i])
    thresholds.append(current)
new["門檻"] = thresholds

print("== 1. 門檻跟著資料跑 ==")
print(f"4 週共 {len(new)} 筆，校準 {len(new) // EVERY} 次，出現過 {new.門檻.nunique()} 種門檻")
print(f"最低 {new.門檻.min()}、最高 {new.門檻.max()}")
weekly = new.groupby((new.event_time - pd.Timestamp("2026-01-12", tz="Asia/Taipei")).dt.days // 7 + 1).門檻.agg(["min", "max"])
print(weekly.rename_axis("週").to_string())

print("\n== 2. 同一筆事件，換個時間處理，結果就不同 ==")
final = new.門檻.iloc[-1]
flip = new[(new.confidence >= new.門檻) != (new.confidence >= final)]
print(f"用「當時的門檻」和「最後的門檻 {final}」判斷結果不同的事件：{len(flip)} 筆")
print(flip[["event_id", "event_time", "site", "confidence", "門檻"]].head(3).to_string(index=False))

print("\n== 3. C 工地換攝影機之後，這個門檻抓到幾成？ ==")
relabel = pd.read_csv(Path(__file__).resolve().parent / "relabel-C.csv")
c = new.drop(columns="needs_notify").merge(relabel[["event_id", "needs_notify"]], on="event_id")
print(f"C 工地 01-26 之後 {len(c)} 筆（重新標註後，需要通知 {c.needs_notify.sum()} 筆）")
print(score_row("自動校準門檻", c.needs_notify, (c.confidence >= c.門檻).astype(int)))
print(f"C 工地 01-26 之後最高 confidence：{c.confidence.max()}；高於當時門檻的：{(c.confidence >= c.門檻).sum()} 筆")
