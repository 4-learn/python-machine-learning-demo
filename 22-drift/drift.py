"""22｜監控的小工具：PSI、分位數比較、找變化點。23 節沿用。"""
from __future__ import annotations

import numpy as np
import pandas as pd

# confidence 的固定分箱：0.4 以下、每 0.1 一格、0.8 以上
BINS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]


def shares(values, bins=BINS) -> np.ndarray:
    """每一格占幾成。空格給 0.001，避免 log(0)。"""
    counts = np.histogram(values, bins)[0]
    return np.maximum(counts / max(len(values), 1), 0.001)


def psi(reference, current, bins=BINS) -> float:
    """Population Stability Index：每一格「現在占比 − 當初占比」×「兩者比值的 log」，加總。
    0 表示分布一樣；差越多越大。"""
    a, b = shares(reference, bins), shares(current, bins)
    return float(((b - a) * np.log(b / a)).sum())


def decile_bins(reference) -> np.ndarray:
    """其他欄位沒有現成的分箱：用訓練期的十分位數切，頭尾開放。"""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def quantiles(values, qs=(0.25, 0.5, 0.75, 0.9)) -> pd.Series:
    return pd.Series(np.quantile(values, qs), index=[f"p{int(q * 100)}" for q in qs])


def ks(a, b) -> float:
    """兩段資料的累積分布曲線，最大差多少（0～1）。"""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                                - np.searchsorted(b, grid, side="right") / len(b))))


def change_point(df: pd.DataFrame, column: str = "confidence", min_days: int = 3):
    """逐日試切：每個候選日期把資料分成「之前」與「之後」，找兩段差最多的那一天。
    回傳 (日期, 差距, 之前筆數, 之後筆數)。前後各至少 min_days 天。"""
    day = df.event_time.dt.strftime("%m-%d")
    days = sorted(day.unique())
    best = None
    for d in days[min_days:len(days) - min_days + 1]:
        before, after = df[day < d][column], df[day >= d][column]
        gap = ks(before, after)
        if best is None or gap > best[1]:
            best = (d, round(gap, 3), len(before), len(after))
    return best



def normal_swing(history: pd.DataFrame, column: str = "confidence", start: str = "2025-11-03") -> float:
    """「沒出事」時 PSI 會晃多大：訓練期每一週和整段比，取最大值。
    不同工地、不同筆數晃動不同，所以每個工地各算一把尺。"""
    week = (history.event_time - pd.Timestamp(start, tz="Asia/Taipei")).dt.days // 7
    return float(max(psi(history[column], g[column]) for _, g in history.groupby(week)))
