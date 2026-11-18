"""34｜pandas quantile、NumPy histogram：分布有沒有變（分位數、自己寫 PSI、每週分組、不要自動改門檻）。"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
# 上線前：一支攝影機的信心值；上線後兩種情況：沒變、攝影機被移動之後整體變低
before = pd.Series(np.clip(rng.normal(0.63, 0.12, 2000), 0.05, 0.99))
same = pd.Series(np.clip(rng.normal(0.63, 0.12, 500), 0.05, 0.99))
moved = pd.Series(np.clip(rng.normal(0.45, 0.12, 500), 0.05, 0.99))

print("== 1. quantile：把分布攤成幾個數字 ==")
q = [0.1, 0.25, 0.5, 0.75, 0.9]
table = pd.DataFrame({"上線前": before.quantile(q), "之後（沒變）": same.quantile(q),
                      "之後（攝影機被移動）": moved.quantile(q)}).T.round(3)
table.columns = ["p10", "p25", "p50", "p75", "p90"]
print(table.to_string())

print("\n== 2. PSI：用上線前的十分位數切 10 格，比每格占幾成 ==")
edges = np.quantile(before, np.linspace(0, 1, 11))
edges[0], edges[-1] = -np.inf, np.inf
share = pd.DataFrame({name: np.histogram(s, edges)[0] / len(s)
                      for name, s in [("上線前", before), ("沒變", same), ("被移動", moved)]})
share.index = [f"第{i + 1}格" for i in range(10)]
print(share.T.round(3).to_string())


def psi(expected, actual, edges):
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


print(f"PSI（沒變）  ：{psi(before, same, edges):.3f}")
print(f"PSI（被移動）：{psi(before, moved, edges):.3f}")

print("\n== 3. pd.Grouper(freq='W-SUN')：每週（週一到週日）一組 ==")
days = pd.date_range("2026-01-19", "2026-02-02", freq="1D")      # 週一 01-19 到下下週一 02-02
daily = pd.DataFrame({"ts": days, "median_conf": [0.63] * 7 + [0.45] * 7 + [0.55]})
weekly = daily.groupby(pd.Grouper(key="ts", freq="W-SUN"))["median_conf"].agg(["count", "min", "max"])
print(weekly.to_string())

print("\n== 4. 不要讓門檻自己跟著分布跑 ==")
high = 0.8
print(f"門檻 {high}：上線前自動叫人 {(before >= high).mean():.3f}，被移動之後 {(moved >= high).mean():.3f}")
auto = moved.quantile(1 - (before >= high).mean())
print(f"「自動校準」：為了維持同樣的叫人比例，門檻會被拉到 {auto:.3f}")
print("但信心值變低，可能是畫面變了、偵測器看錯了，也可能是現場真的變安全了；沒有標籤，分不出來")
