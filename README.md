# Python 機器學習（2026 v2）— 範例程式與進度存檔

這門課替一個告警平台做新零件：**告警判斷器**。它讀告警的資料，決定這一筆要 `auto_alert`（馬上通知）、`require_review`（待人確認）還是 `log_only`（只記錄）。

## 兩條線，一份程式

| | 工廠線 | 健康線 |
| --- | --- | --- |
| 客戶 | 金屬加工廠 | 護理之家 |
| 事件 | `no_helmet`、`aisle_blocked` | `fall_detected`、`bed_exit` |
| 誰來處理 | 值班主管 | 值班護理師 |

開課時選一條線，整門課用 `LINE=factory` 或 `LINE=care`。

## 內容

| 路徑 | 說明 |
| --- | --- |
| `demo/NN-*/demo.py` | 講解用的小例子，一節教一個工具。在 repo 根目錄執行：`python demo/01-crosstab/demo.py` |
| `story/vNN-*/judge.py` | 進度存檔：做完第 NN 節的判斷器。在該資料夾執行：`LINE=factory python judge.py` |
| `story/profiles/` | 客戶設定檔（多了 ML 用的欄位） |
| `story/data/` | 平台 10 週的告警歷史（**合成資料**，`tools/build_story_data.py` 產生，`--check` 可驗證） |

學員請在 `story/` 底下開自己的資料夾（例如 `story/my/`），整門課在裡面寫 `judge.py`。

## 環境

Python 3.11 以上，CPU 即可，不需要顯示卡。

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows：.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 已經指定從 PyTorch 官方的 **CPU 版**套件庫下載 `torch`、`torchvision`（第 17 節說明原因：預設版本含顯示卡套件，會多佔好幾 GB）。所有版本都已固定，和講義的實測輸出一致。
