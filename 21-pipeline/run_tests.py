"""21｜執行 test_pipeline.py，只印出每個測試的結果（不印時間，方便貼進講義）。

    python 21-pipeline/run_tests.py        # 等同 python -m pytest 21-pipeline -v
"""
import re
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "pytest", "21-pipeline/test_pipeline.py", "-v", "-p", "no:cacheprovider"],
                   capture_output=True, text=True)
print("== 7. pytest 驗收 ==")
for line in r.stdout.splitlines():
    m = re.match(r".*::(\w+) (PASSED|FAILED|ERROR)", line)
    if m:
        print(f"{m.group(2):7s} {m.group(1)}")
print("全部通過：", r.returncode == 0)
