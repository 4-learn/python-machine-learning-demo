"""23｜端到端驗收：照驗收標準（acceptance.yaml）逐條量，結果寫進 acceptance-result.csv。

    python 23-acceptance/accept.py       # CPU 約 30 秒（含 YOLO 原圖與夜間模擬各 90 張）
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "19-detection-eval"))
sys.path.insert(0, str(HERE.parent / "22-drift"))
from system import DetectionPipeline, load_frames  # noqa: E402
from detlib import HEAD, detect, images_of, sweep  # noqa: E402
from drift import normal_swing, psi  # noqa: E402

from common import DATA, load_events, score_row, time_split  # noqa: E402

spec = yaml.safe_load((HERE / "acceptance.yaml").read_text(encoding="utf-8"))
checks = {c["id"]: c for c in spec["checks"]}
log = yaml.safe_load((HERE / "test-log.yaml").read_text(encoding="utf-8"))["openings"]
results = []


def record(cid, value, ok):
    results.append({"id": cid, "項目": checks[cid]["name"], "標準": checks[cid].get("min", checks[cid].get("max", "")),
                    "實測": value, "結果": "通過" if ok else "不通過"})


def alerts(decisions):
    """政策層級的「有沒有叫到人」：notify 與被冷卻併入的 merged 都算。"""
    return decisions.action.isin(["notify", "merged"]).astype(int)


print("== 1. 驗收標準（打開前寫好，版本", spec["version"], "）==")
for c in spec["checks"]:
    bound = f"≥ {c['min']}" if "min" in c else f"≤ {c['max']}" if "max" in c else c.get("rule", "見說明")
    print(f"{c['id']}  {c['name']:28s} {bound}")
print("打開前的預期（valid）：recall", spec["expected"]["recall"], "每天通知", spec["expected"]["notify_per_day"])

print("\n== 2. 五個 class 串起來：90 張新照片 → 決定（Mock 偵測）==")
system = DetectionPipeline(mock=True)
print("門檻來源：", system.calibrator.value, system.calibrator.provenance["approved_by"],
      system.calibrator.provenance["approved_at"])
batch = system.process_frames(load_frames())
print(f"90 張圖 → {len(batch)} 筆事件；", batch.action.value_counts().to_dict())
again = DetectionPipeline(mock=True).process_frames(load_frames())
print("重跑一次，結果逐欄相同：", batch.equals(again))

print("\n== 3. 偵測（scenes 測試集 90 張，門檻 0.25）==")
from ultralytics import YOLO  # noqa: E402

model, test_images = YOLO(system.detector.weights), images_of("test")
for cid, night in [("D1", False), ("D2", True)]:
    row = sweep(detect(model, test_images, night), test_images, HEAD, [system.detector.min_confidence])[0]
    print(f"{'夜間模擬' if night else '原圖':6s} TP {row['TP']}  FP {row['FP']}  FN {row['FN']}  recall {row['recall']}")
    record(cid, row["recall"], row["recall"] >= checks[cid]["min"])

print("\n== 4. test 第二次打開 ==")
assert [o["lesson"] for o in log] == ["10", "23"], "test-log.yaml 沒有記錄這次開封，先記錄再打開"
print("開封紀錄：", " → ".join(f"{o['lesson']} 節（{o['date']}）" for o in log))
_, test = time_split(load_events())
t = system.engine.pipe.score_table(test)
y = test.needs_notify.astype(int)
rule = (test.confidence >= 0.7).astype(int)
table = pd.DataFrame([score_row("現行規則 conf≥0.7", y, rule), score_row("管線（模型＋規則＋冷卻）", y, alerts(t))])
print(table.to_string(index=False))
per_day = t[t.action == "notify"].groupby(t.event_time.dt.date).size()
rest = int(((t.zone == "休息區") & (t.action == "notify")).sum())
print(f"每天通知 平均 {per_day.sum() / 14:.1f} 次、最多 {per_day.max()} 次；休息區通知 {rest} 次")
record("T1", table.recall.iloc[1], table.recall.iloc[1] >= checks["T1"]["min"])
record("T2", round(per_day.sum() / 14, 1), per_day.sum() / 14 <= checks["T2"]["max"])
record("T3", rest, rest <= checks["T3"]["max"])

print("\n== 5. 上線 4 週（已標註 463 筆）==")
new = load_events(DATA / "events-new-weeks.csv")
labeled = new.dropna(subset=["needs_notify"])
n = system.engine.pipe.score_table(labeled)
y = labeled.needs_notify.astype(int)
rule = (labeled.confidence >= 0.7).astype(int)


def cost(pred):
    return int(10 * ((y == 1) & (pred == 0)).sum() + ((y == 0) & (pred == 1)).sum())


table = pd.DataFrame([{**score_row("現行規則 conf≥0.7", y, rule), "cost": cost(rule)},
                      {**score_row("管線（模型＋規則＋冷卻）", y, alerts(n)), "cost": cost(alerts(n))}])
print(table.to_string(index=False))
print(f"該通知 {y.sum()} 筆（test 是 {test.needs_notify.sum():.0f} 筆）：多漏 1 筆，recall 就少 {1 / y.sum():.3f}")
record("N1", table.recall.iloc[1], table.recall.iloc[1] >= checks["N1"]["min"])
record("N2", f"{table.cost.iloc[1]} vs {table.cost.iloc[0]}", table.cost.iloc[1] < table.cost.iloc[0])

print("\n== 6. 錯誤分析：上線 4 週的漏報 ==")


def reason(r):
    if r.zone == "休息區" or r.duration_sec < 20:
        return "標籤違反標註規則"
    if r.proba >= 0.1:
        return "接近門檻（0.1～0.2）"
    if r.light_lux < 300:
        return "夜間低照度"
    return "其他"


missed = n[(y == 1) & (alerts(n) == 0)]
reasons = missed.apply(reason, axis=1)
print(reasons.value_counts().rename_axis("原因").rename("筆數").to_string())
fixed = y.where(~((y == 1) & (n.zone.eq("休息區") | (n.duration_sec < 20))), 0)
print(f"漏報 {len(missed)} 筆；把違反標註規則的標籤改成 0 後，recall {score_row('', fixed, alerts(n))['recall']}")
print("該通知的事件 prior_violations_30d 平均：test", round(test[test.needs_notify == 1].prior_violations_30d.mean(), 2),
      "／上線", round(labeled[y == 1].prior_violations_30d.mean(), 2))
missed.assign(原因=reasons)[["event_id", "site", "zone", "confidence", "duration_sec", "light_lux",
                           "prior_violations_30d", "proba", "原因"]].to_csv(HERE / "missed-new-weeks.csv", index=False)

print("\n== 7. 漂移（各工地每週 confidence PSI 最大值 vs 訓練期正常晃動）==")
train, _ = time_split(load_events())
week = (new.event_time - pd.Timestamp("2026-01-12", tz="Asia/Taipei")).dt.days // 7
rows = []
for site, g in new.groupby("site"):
    ref = train[train.site == site]
    worst = max(psi(ref.confidence, w.confidence) for _, w in g.groupby(week[g.index]))
    rows.append({"site": site, "上線最大 PSI": round(worst, 3), "正常晃動": round(normal_swing(ref), 3)})
drift = pd.DataFrame(rows).assign(超出=lambda d: d["上線最大 PSI"] > d["正常晃動"])
print(drift.to_string(index=False))
record("M1", "超出：" + ("、".join(drift[drift.超出].site) or "無"), not drift.超出.any())

print("\n== 8. 驗收結果 ==")
result = pd.DataFrame(results)
print(result.to_string(index=False))
result.to_csv(HERE / "acceptance-result.csv", index=False)
print(f"通過 {(result.結果 == '通過').sum()} / {len(result)}；結論：", "通過" if (result.結果 == "通過").all() else "不通過，寫入報告，不改模型")
