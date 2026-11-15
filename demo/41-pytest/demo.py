"""41｜pytest：判斷器的自動驗收。不測模型準不準，測它該守的規則。"""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toy_judge import ToyJudge, train  # noqa: E402

print("== 1. 被測的程式：一個很小的判斷器 ==")
judge = ToyJudge(train())
for conf, dur, rest in [(0.95, 70.0, 0), (0.95, 8.0, 1), (0.6, 40.0, 0), (0.3, 5.0, 0)]:
    print(f"confidence {conf}、{dur} 秒、休息區 {rest} → score {judge.score(conf, dur, rest):.3f}、"
          f"{judge.decide(conf, dur, rest)}")

print("\n== 2. assert：條件成立就安靜，不成立就丟出 AssertionError ==")
assert judge.score(0.99, 30.0) >= judge.score(0.31, 30.0)
print("信心值越高分數不降：通過（什麼都沒印）")
try:
    assert judge.decide(0.95, 8.0, rest_area=1) == "auto_alert", "休息區也自動叫？"
except AssertionError as e:
    print(f"AssertionError：{e}")


def run_pytest(*args, env=None):
    """在這個資料夾跑 pytest，印出結果（拿掉每次都會變的秒數和環境資訊）。"""
    r = subprocess.run([sys.executable, "-m", "pytest", "test_toy_judge.py", "-p", "no:cacheprovider", *args],
                       cwd=HERE, capture_output=True, text=True, env={**os.environ, "COLUMNS": "80", **(env or {})})
    lines = [re.sub(r" in [0-9.]+s|\s+\[\s*\d+%\]$", "", l).strip("= ") for l in r.stdout.splitlines()]
    return r.returncode, [l for l in lines if l.strip() and not l.startswith(("platform", "rootdir", "cachedir", "plugins", "configfile"))]


print("\n== 3. pytest -v：找出所有 test_ 開頭的函式，一個一個跑 ==")
code, out = run_pytest("-v")
print("\n".join(l for l in out if "::" in l or "passed" in l or "failed" in l))
print(f"結束代碼：{code}（0＝全部通過）")

print("\n== 4. fixture(scope='module')：判斷器只訓練一次，6 個測試共用 ==")
code, out = run_pytest("-q", "--setup-show")
print("\n".join(l.strip() for l in out if "SETUP" in l or "TEARDOWN M" in l))

print("\n== 5. 有人把判斷器「簡化」成只看信心值：測試抓到 ==")
code, out = run_pytest("-q", "--tb=no", "-rf", env={"TOY_BROKEN": "1"})
print("\n".join(l for l in out if l.startswith("FAILED") or "passed" in l))
print(f"結束代碼：{code}（不是 0，CI 會擋下來）")
