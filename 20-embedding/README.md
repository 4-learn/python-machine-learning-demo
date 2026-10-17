# 20｜Embedding 與「何時用 LLM」

巡檢回報短文字（`data/reports.csv`）要不要通知：關鍵字規則、字元 TF-IDF＋邏輯迴歸、`BAAI/bge-small-zh-v1.5`（revision 固定）embedding＋邏輯迴歸，比較準確度、成本、延遲、可重現。`new_reports.csv` 是 20 句換了說法的回報（合成），用來檢查「沒看過的說法」。

    HF_HUB_OFFLINE=1 python 20-embedding/compare.py

模型須先下載到本機快取（與 LLM 課 16 節同一個 revision）。
