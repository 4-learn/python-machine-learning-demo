"""20｜ultralytics 的 train：用 20 張場景照片親手訓練 3 輪，看流程和產出的檔案（模型很爛是正常的）。

執行（在 repo 根目錄）：
    python demo/20-yolo/train_tiny.py            # 工廠線
    LINE=care python demo/20-yolo/train_tiny.py  # 健康線
"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # 不上網檢查更新

import shutil
import tempfile
import time
from pathlib import Path

from ultralytics import YOLO

LINE = os.getenv("LINE", "factory")
ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "story" / "data" / LINE / "scenes"
OUT = Path(__file__).resolve().parent / "out"
NAMES = {"factory": ["helmet", "head"], "care": ["person", "fallen"]}[LINE]

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    print(f"== 1. 準備小資料集：{LINE} 的 train 前 20 張、val 前 10 張 ==")
    for split, n in [("train", 20), ("val", 10)]:
        (tmp / "images" / split).mkdir(parents=True)
        (tmp / "labels" / split).mkdir(parents=True)
        for img in sorted((SCENES / "images" / split).glob("*.jpg"))[:n]:
            shutil.copy(img, tmp / "images" / split / img.name)
            shutil.copy(SCENES / "labels" / split / f"{img.stem}.txt", tmp / "labels" / split / f"{img.stem}.txt")
        print(f"{split}：{len(list((tmp / 'images' / split).glob('*.jpg')))} 張圖、"
              f"{len(list((tmp / 'labels' / split).glob('*.txt')))} 個標註檔")
    yaml_text = (f"path: {tmp}\ntrain: images/train\nval: images/val\nnames:\n"
                 + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    (tmp / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print("data.yaml：")
    print(yaml_text.replace(str(tmp), "（暫存資料夾）").rstrip())

    print("\n== 2. model.train：從 COCO 預訓練的 yolo11n 開始，訓練 3 輪 ==")
    start = time.perf_counter()
    model = YOLO(str(ROOT / "models" / "yolo11n.pt"))
    metrics = model.train(data=str(tmp / "data.yaml"), epochs=3, imgsz=320, batch=8, device="cpu", workers=0,
                          seed=0, deterministic=True, verbose=False, plots=False,
                          project=str(OUT), name=f"tiny-{LINE}", exist_ok=True)
    print(f"花了 {time.perf_counter() - start:.0f} 秒（每台電腦不一樣）")
    map50 = metrics.results_dict["metrics/mAP50(B)"]
    print(f"val 10 張的 mAP50：{map50:.2f}（教師用 240 張、30 輪的模型是 0.995）")
    print(f"小模型的 mAP50 低於 0.5：{map50 < 0.5}")

run = OUT / f"tiny-{LINE}"
print("\n== 3. 產出了哪些檔案 ==")
for path in sorted(p for p in run.rglob("*") if p.is_file()):
    print(path.relative_to(run))

print("\n== 4. 讀回 best.pt，看它學到什麼 ==")
tiny = YOLO(str(run / "weights" / "best.pt"))
print(f"類別：{tiny.names}")
image = sorted((SCENES / "images" / "test").glob("*.jpg"))[0]
for name, m in [("3 輪小模型", tiny), ("教師的模型", YOLO(str(ROOT / "models" / f"{LINE}-yolo11n.pt")))]:
    boxes = m.predict(str(image), conf=0.25, imgsz=320, device="cpu", verbose=False)[0].boxes
    print(f"{name}：{image.name} 上信心值 0.25 以上的框 {len(boxes)} 個")
