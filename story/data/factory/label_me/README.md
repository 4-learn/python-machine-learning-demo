# story/data/factory/label_me：自己畫框

這 10 張是工廠的合成照片（和 `scenes/images/train` 的前 10 張相同）。
用 Label Studio 替每個人畫框：有戴安全帽的頭畫 `helmet`，沒戴的頭畫 `head`。
標準答案在 `scenes/labels/train/` 同名的 `.txt`，畫完用 `demo/20-yolo/check_labels.py` 對答案。

| 檔案 | 做什麼 |
| --- | --- |
| `images/train_000.jpg`～`train_009.jpg` | 要標註的 10 張照片（320×320） |
| `label_config.xml` | Label Studio 的標註設定：兩個類別 helmet、head |

## 步驟

1. 另開一個環境安裝 Label Studio（約 800 MB，第一次才要）：

   ```bash
   python3 -m venv .venv-label
   source .venv-label/bin/activate        # Windows：.venv-label\Scripts\Activate.ps1
   pip install -r requirements-label.txt
   ```

2. 啟動，瀏覽器打開 http://localhost:8080 ，註冊一個帳號（只存在你自己的電腦）：

   ```bash
   label-studio start
   ```

3. Create Project → 名稱「安全帽標註」→ Data Import 拖進 `images/` 的 10 張 →
   Labeling Setup 選 Custom template，貼上 `label_config.xml` 的內容 → Save。

4. 每張照片：先按類別（或按鍵 1、2），再在頭上拖一個框，框要貼著頭；全部畫完按 Submit。

5. Export → 選 **YOLO with Images**（只選 YOLO 會沒有照片）→ 下載 zip，
   解壓到 `story/my/labels-export/`。

6. 回到課程環境（`.venv`），對答案、整理成訓練用的資料夾：

   ```bash
   python demo/20-yolo/check_labels.py story/my/labels-export
   python demo/20-yolo/train_mine.py
   ```

## 要注意

- **類別編號會被重排**：Label Studio 照字母排，匯出是 `0 head`、`1 helmet`，和課程的偵測器（`0 helmet`、`1 head`）相反。
  `check_labels.py` 會讀 `classes.txt`，自動換回課程的順序。
- **匯出的檔名前面多一段亂碼**（例如 `3791a83a-train_004.jpg`）：照片和標註一起加，不影響訓練。
- **Label Studio 沒有產生 `data.yaml`**：`check_labels.py` 會幫你寫。
