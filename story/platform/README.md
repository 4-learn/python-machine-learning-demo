# story/platform：自己架的告警平台（FastAPI）

12 之前，偵測器都送到程式裡自己開的「替身平台」（`story/platform_mock.py`）。
這個資料夾是同樣規則的真正 server，用 FastAPI 寫成，要另外啟動。

## 規則

| 情況 | 平台回答 |
|---|---|
| 沒帶正確的 `X-API-Key`（`dev-device`） | 401 |
| 事件類型不在客戶設定檔（`story/profiles/factory.json`） | 400 |
| 信心值 ≥ 0.8 | 200，開告警 `open`（馬上通知） |
| 信心值 0.5～0.8 | 200，開告警 `pending_review`（待確認） |
| 信心值 < 0.5 | 200，只記錄，不開告警 |
| 事件的 `detail.judge` 有判斷器的決定（13） | 照判斷器分流 |

## 怎麼跑：開兩個終端機

```bash
# 終端機 1：啟動平台（在 repo 根目錄）
cd story/platform
uvicorn main:app --port 8000

# 終端機 2：偵測器送到自己的平台（在 repo 根目錄）
cd story/my
PLATFORM_URL=http://localhost:8000 LINE=factory python detector.py
```

瀏覽器打開：

- http://localhost:8000/alerts ：目前開了哪些告警
- http://localhost:8000/docs ：FastAPI 自動產生的 API 說明頁，可以直接在網頁上送事件試試

停止平台：在終端機 1 按 `Ctrl+C`。沒有資料庫，停掉後告警就不見。
