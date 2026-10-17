"""24｜換成 LLM 課學生：只拿到 handoff/，沒有 common.py、沒有模型。
寫一個新方法（照標註規則寫的規則），在 dev-sample 上定參數，交給 score.py 評分；再把一列轉成文字。

    python 24-handoff/consumer.py         # 先跑 build.py
"""
import csv
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent / "handoff"


def read(name):
    with open(PKG / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rule(row, cut):
    """照 LABELING.md 的三條寫：不在休息區、持續 25 秒以上、偵測器信心夠高（代替「真的沒戴」）。"""
    return int(row["zone"] != "休息區" and float(row["duration_sec"]) >= 25 and float(row["confidence"]) >= cut)


def cost(rows, cut):
    return sum(10 if int(r["needs_notify"]) and not rule(r, cut) else 1 if rule(r, cut) and not int(r["needs_notify"])
               else 0 for r in rows)


print("== 1. 在 dev-sample（40 筆）上挑信心門檻，不碰評測集 ==")
dev = read("dev-sample.csv")
for cut in [0.4, 0.5, 0.6, 0.7]:
    print(f"信心 ≥ {cut}：dev 成本 {cost(dev, cut)}")
best = min([0.4, 0.5, 0.6, 0.7], key=lambda c: (cost(dev, c), -c))
print("選定：", best)

print("\n== 2. 用選定的版本產生預測，交給 score.py ==")
out = PKG.parent / "標註規則版.csv"
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["event_id", "pred"])
    w.writerows([r["event_id"], rule(r, best)] for r in read("eval-set.csv"))
r = subprocess.run([sys.executable, "-I", "score.py", str(out)], cwd=PKG, capture_output=True, text=True)
print(r.stdout.strip())
out.unlink()

print("\n== 3. 給 LLM 的版本：表格的一列 → 一段文字 ==")
row = read("dev-sample.csv")[0]
text = (f"{row['site']} 工地{row['zone']}，{row['shift']} {row['hour']} 點。偵測器懷疑有人沒戴安全帽，"
        f"信心 {row['confidence']}，持續 {row['duration_sec']} 秒，照度 {row['light_lux']} lux，"
        f"附近 {row['people_nearby']} 人。這位工人年資 {row['tenure_days']} 天，過去 30 天被通知 {row['prior_violations_30d']} 次。")
print(text)
print("標籤：", row["needs_notify"])
