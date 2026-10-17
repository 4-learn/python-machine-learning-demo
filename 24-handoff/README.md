# 24｜交給 LLM 課：凍結評測集與 baseline

把 test 期間的 977 筆凍結成交接包 `handoff/`，附上指紋、標註規則、baseline 成績表、門檻設定、已知錯誤與只用標準函式庫的評分程式。LLM 課（01／07／16／17）用同一份評測集、同一把尺比較。

    python 24-handoff/build.py       # 產出 handoff/ 並做結構檢查
    python 24-handoff/consumer.py    # 換成 LLM 課學生：只用 handoff/ 寫新方法、評分、把一列轉成文字

| 檔案 | 內容 |
| --- | --- |
| `handoff.py` | `build_package`、`check_package`、`eval_set`、`dev_sample`、`predictions`、`scores`、`known_errors` |
| `template/` | 交接包裡不會變的檔案：`README.md`、`LABELING.md`、`score.py` |
| `handoff/` | `build.py` 的產出（就是要交出去的那包） |

交接包內容：`eval-set.csv`（977 筆，欄位＝`event_id`、`incident_id`、FEATURES、`needs_notify`）、`eval-set.sha256`、`dev-sample.csv`（驗證期 40 筆）、`LABELING.md`、`scores.csv`、`notify.yaml`（`config/notify.yaml` 副本）、`known-errors.csv`、`manifest.json`、`score.py`、`README.md`。
