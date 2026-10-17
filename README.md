# Python 機器學習實務（2026）— 範例程式

勞動部 AI 大數據人才養成班「Python 機器學習實務」24 節課的範例程式與資料。講義在 HackMD：<https://hackmd.io/@yillkid/S1MV44-sMl>

全課只回答一個問題：**這個「疑似未戴安全帽」事件，要不要通知現場主管？**

## 環境（Python 3.11 以上，CPU 即可）

```bash
python3.11 -m venv ~/.venvs/ml-course
source ~/.venvs/ml-course/bin/activate
# 先裝 CPU 版 PyTorch（預設版本含 CUDA，venv 會從 0.9 GB 變 5.6 GB）
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn==1.9.1 pandas==3.0.6 matplotlib pytest pillow joblib pyyaml sentence-transformers ultralytics
pip install -e .          # 讓各節程式能 from common import ...
```

所有程式都從 repo 根目錄執行，例如 `python 05-knn/knn_first.py`。

## 內容

| 路徑 | 說明 |
| --- | --- |
| `common.py` | 全課共用：讀資料、特徵清單、切割、成績表 |
| `data/` | **合成**資料（`tools/build_data.py` 產生，`MANIFEST.json` 有 SHA-256；`--check` 可驗證） |
| `models/` | `helmet-yolo11n.pt`（教師以合成場景訓練）、`yolo11n.pt`（Ultralytics COCO 預訓練）、`notify-hgb.joblib` |
| `config/notify.yaml` | 已核准的通知門檻與模型版本（`tools/build_artifacts.py` 產生） |
| `NN-*/` | 各節範例 |

## 授權與資料說明

- 資料全部由程式合成，人員、工地、事件、文字都不是真實資料；`data/photo-*.jpg` 為 AI 生成圖片。合成資料上的分數**不代表**真實工地的表現。
- `ultralytics` 與 YOLO11 權重為 **AGPL-3.0**：若把使用它的服務提供給他人，須依 AGPL 公開對應原始碼，或另購商業授權。
- torchvision 預訓練權重依其模型卡授權。
