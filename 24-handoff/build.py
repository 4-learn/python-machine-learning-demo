"""24｜產出交接包 24-handoff/handoff/，再用 LLM 課學生的角度驗一次。

    python 24-handoff/build.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from handoff import EVAL_COLUMNS, FILES, build_package, check_package, sha256  # noqa: E402

OUT = HERE / "handoff"

print("== 1. 凍結評測集 ==")
m = build_package(OUT)
e = m["eval_set"]
print(f"{m['dataset_version']}／{m['annotation_version']}：{e['rows']} 筆、{e['incidents']} 個 incident、"
      f"要通知 {e['positives']} 筆（{e['period']}）")
print("欄位：", "、".join(EVAL_COLUMNS))
print("sha256：", e["sha256"])

print("\n== 2. 指紋：同樣的資料，寫兩次一樣嗎？==")
build_package(HERE / "_again")
print("重新產生一次，指紋相同：", sha256(HERE / "_again" / "eval-set.csv") == e["sha256"])
ev = pd.read_csv(OUT / "eval-set.csv")
ev.loc[0, "needs_notify"] = 1 - ev.loc[0, "needs_notify"]
ev.to_csv(HERE / "_again" / "eval-set.csv", index=False, lineterminator="\n")
print("只改 1 筆標籤，指紋變成：", sha256(HERE / "_again" / "eval-set.csv")[:12])
pd.read_csv(OUT / "eval-set.csv").to_csv(HERE / "_again" / "eval-set.csv", index=False, encoding="utf-8-sig")
print("內容不改、用 Excel 習慣的 utf-8-sig 存一次，指紋：", sha256(HERE / "_again" / "eval-set.csv")[:12])
pd.read_csv(OUT / "eval-set.csv").sort_values("site", kind="stable").to_csv(
    HERE / "_again" / "eval-set.csv", index=False, lineterminator="\n")
print("內容不改、只換列的順序，指紋：", sha256(HERE / "_again" / "eval-set.csv")[:12])

print("\n== 3. 成績表（baseline）==")
print(pd.read_csv(OUT / "scores.csv").drop(columns=["n", "dataset_version"]).to_string(index=False))

print("\n== 4. 已知錯誤（13 節同一套分類，定版模型在評測集上）==")
k = pd.read_csv(OUT / "known-errors.csv")
print(k.groupby(["錯誤", "原因"]).size().rename("筆數").reset_index().to_string(index=False))

print("\n== 5. 交接包內容 ==")
for f in FILES:
    print(f"{f:18s} {(OUT / f).stat().st_size:>7,d} bytes")

print("\n== 6. 用 LLM 課學生的角度：只有 Python 標準函式庫，跑 score.py ==")
r = subprocess.run([sys.executable, "-I", "score.py"], cwd=OUT, capture_output=True, text=True)
print(r.stdout.strip())

print("\n== 7. 結構檢查 ==")
print(check_package(OUT).to_string(index=False))

import shutil  # noqa: E402

shutil.rmtree(HERE / "_again")
