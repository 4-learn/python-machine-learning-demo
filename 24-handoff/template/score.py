"""用同一把尺評分：只用 Python 標準函式庫（不需要 pandas、sklearn）。

    python score.py                       # 檢查指紋、印出成績表
    python score.py my-predictions.csv    # 加上你的方法（欄位：event_id,pred；pred 為 0 或 1）

指紋不符時拒絕評分：評測集被改過，成績就不能和 scores.csv 比。
"""
import csv
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FN_COST, FP_COST = 10, 1


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score(name, gold, pred):
    tp = sum(1 for k, y in gold.items() if y == 1 and pred[k] == 1)
    fp = sum(1 for k, y in gold.items() if y == 0 and pred[k] == 1)
    fn = sum(1 for k, y in gold.items() if y == 1 and pred[k] == 0)
    n = len(gold)
    return {"model": name, "accuracy": round((n - fp - fn) / n, 3),
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3), "alerts": tp + fp, "cost": FN_COST * fn + FP_COST * fp}


def main(argv):
    recorded = (HERE / "eval-set.sha256").read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256((HERE / "eval-set.csv").read_bytes()).hexdigest()
    if recorded != actual:
        sys.exit(f"指紋不符：記錄 {recorded[:12]}，實際 {actual[:12]}。評測集被改過，拒絕評分。")
    gold = {r["event_id"]: int(r["needs_notify"]) for r in read(HERE / "eval-set.csv")}
    print(f"評測集 {len(gold)} 筆，指紋 {actual[:12]} 相符")
    rows = [{k: r[k] for k in ["model", "accuracy", "precision", "recall", "alerts", "cost"]}
            for r in read(HERE / "scores.csv")]
    for path in argv:
        pred = {r["event_id"]: int(r["pred"]) for r in read(path)}
        lacking = set(gold) - set(pred)
        if lacking:
            sys.exit(f"{path} 少了 {len(lacking)} 筆事件的預測；每一筆都要回答（不會就答 1，交給人看）")
        rows.append(score(Path(path).stem, gold, pred))
    print("accuracy  precision  recall  alerts  cost  model")
    for r in rows:
        print(f"{float(r['accuracy']):8.3f}  {float(r['precision']):9.3f}  {float(r['recall']):6.3f}  "
              f"{int(r['alerts']):6d}  {int(r['cost']):4d}  {r['model']}")


if __name__ == "__main__":
    main(sys.argv[1:])
