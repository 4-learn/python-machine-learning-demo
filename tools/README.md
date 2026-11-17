# tools/：教師用工具

學員不用執行這裡的程式；資料與模型都已經放在 repo 裡。

| 檔案 | 做什麼 | 什麼時候用 |
| --- | --- | --- |
| `build_story_data.py` | 產生兩條線 10 週的告警歷史（`story/data/{factory,care}/alerts.csv` 等）| `--check` 確認資料沒被改過 |
| `build_story_images.py` | 產生小圖（`crops/`）與 YOLO 場景（`scenes/`）| 同上 |
| `build_story_extra.py` | 產生 25–42 節的 ROI 畫面、骨架、串流資料 | 同上 |
| `pose_generator.py` | 骨架序列產生器，給 `build_story_extra.py` 呼叫 | 不直接執行 |
| `story/master*.py`、`story/split*.py` | 進度存檔的「單一來源」：一份程式用 `#@vNN` 標記，切成 `story/vNN-*/` 各版 | 改進度存檔時 |

切出進度存檔：

```bash
python3 tools/story/split.py  tools/story/master.py          story   # v01–v15 judge.py
python3 tools/story/split2.py tools/story/master_detector.py story   # v16–v24 detector.py
python3 tools/story/split3.py tools/story/master_extra.py    story   # v25–v40 extra.py
```

v41（pytest）、v42（Workshop 3）是手寫的，不由 master 產生。
