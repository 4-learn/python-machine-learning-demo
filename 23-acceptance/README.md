# 23｜端到端驗收

新一批場景照片（`data/scenes` 測試集 90 張）與上線 4 週事件，跑完整條管線，照事先寫好的驗收標準逐條量，交出可重跑的結果與報告。判斷核心沿用 `21-pipeline/pipeline.py`，漂移檢查沿用 `22-drift/drift.py`，偵測評估沿用 `19-detection-eval/detlib.py`。

    python 23-acceptance/accept.py             # 驗收：逐條量 acceptance.yaml，寫 acceptance-result.csv（約 15 秒）
    python 23-acceptance/run_tests.py          # pytest 23-acceptance（Mock 偵測，不需要 YOLO）
    python 23-acceptance/record_detections.py  # 重新跑 YOLO，更新 detections.json（Mock 用）

| 檔案 | 內容 |
| --- | --- |
| `system.py` | 舊版作業的五個 class：`SafetyDetector`、`EventConverter`、`ThresholdCalibrator`（只讀核准門檻）、`RuleEngine`、`DetectionPipeline` |
| `acceptance.yaml` | 驗收標準：打開 test 與新資料之前寫好，之後不改 |
| `test-log.yaml` | test 的開封紀錄（10 節第一次、23 節第二次） |
| `frames.csv`、`frames-night.csv` | 每張照片的攝影機、時間、追蹤器給的持續秒數與人員（日班／夜班） |
| `detections.json` | 真實 YOLO 在 90 張圖上的 head 框（Mock 模式讀它） |
| `test_system.py` | 驗收測試：重跑相同、門檻是核准的、未核准提案被拒、休息區不通知、test 有開封紀錄 |
