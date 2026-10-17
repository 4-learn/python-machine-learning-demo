# 21｜模型進管線

Event JSON → 補情境欄位 → `models/notify-hgb.joblib` 機率 → `config/notify.yaml` 核准門檻 → `rules.yaml` 政策規則 → 決定（每筆附模型與門檻版本）。`pipeline.py` 可被 import（23 節沿用）；Mock 模式讀 `mock_events.json`（19 節產生的 Event JSON 副本），不載入 YOLO。

    python 21-pipeline/run_mock.py         # Mock 模式：即時路徑＋離線路徑
    python 21-pipeline/run_yolo.py         # 真實模式：YOLO 產生的事件與 Mock 檔相同（CPU 約 10 秒）
    python 21-pipeline/run_tests.py        # 或 python -m pytest 21-pipeline -v

| 檔案 | 內容 |
| --- | --- |
| `pipeline.py` | `NotifyPipeline`、`load_mock_events`、`detect_events`、`summary` |
| `rules.yaml` | 政策規則（由上往下，第一條符合就生效）與冷卻秒數 |
| `site.yaml` | 攝影機 → 工地、區域、各班別照度 |
| `mock_events.json` | Mock 模式的輸入 |
| `test_pipeline.py` | pytest 驗收 |
