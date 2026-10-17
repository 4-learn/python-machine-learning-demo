# 22｜監控與漂移：取代「自動校準」

上線 4 週（`data/events-new-weeks.csv`）的監控：告警率、confidence 分位數、PSI、變化點。發現漂移後重新標註、重新評估、寫提案等人核准——**不自動改門檻**。管線沿用 `21-pipeline/pipeline.py`。

    python 22-drift/monitor.py          # 監控：只看資料，不改設定
    python 22-drift/auto_calibrate.py   # 反例：舊版每 N 筆 KMeans 自動校準
    python 22-drift/reevaluate.py       # 重新標註後重新評估，寫 notify-proposed.yaml

| 檔案 | 內容 |
| --- | --- |
| `drift.py` | `psi`、`quantiles`、`ks`、`change_point`、`normal_swing`（23 節沿用） |
| `relabel-C.csv` | C 工地 2026-01-26～02-08 全部 147 筆重新標註（陳主任，2026-02-10） |
| `notify-proposed.yaml` | `reevaluate.py` 產出的門檻提案（`status: proposed`、未核准）；`config/notify.yaml` 不動 |
