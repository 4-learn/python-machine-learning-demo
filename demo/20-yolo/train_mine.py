"""10｜用自己畫的框訓練：讀 check_labels.py 整理好的 story/my/labels-mine/，訓練 3 輪。

執行（在 repo 根目錄，課程環境 .venv；先跑過 check_labels.py）：
    python demo/20-yolo/train_mine.py

只有 10 張、3 輪，模型很爛是正常的：這一段學的是「自己畫的框，真的能拿去訓練」。
"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # 不上網檢查更新

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "story" / "my" / "labels-mine" / "data.yaml"
OUT = ROOT / "story" / "my" / "runs"

if not DATA.exists():
    raise SystemExit("找不到 story/my/labels-mine/data.yaml：先跑 python demo/20-yolo/check_labels.py 匯出資料夾")

model = YOLO(str(ROOT / "models" / "yolo11n.pt"))
metrics = model.train(data=str(DATA), epochs=3, imgsz=320, batch=8, device="cpu", workers=0,
                      seed=0, deterministic=True, verbose=False, plots=False,
                      project=str(OUT), name="mine", exist_ok=True)
best = YOLO(str(OUT / "mine" / "weights" / "best.pt"))
print(f"\n類別：{best.names}")
print(f"mAP50：{metrics.results_dict['metrics/mAP50(B)']:.2f}（10 張、3 輪，很低是正常的）")
print(f"模型存在：{(OUT / 'mine' / 'weights' / 'best.pt').relative_to(ROOT)}")
