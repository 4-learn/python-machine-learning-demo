"""19｜對照 ultralytics 內建的 model.val()：mAP 是什麼數字。

    python 19-detection-eval/official_val.py

val() 會在標註資料夾旁寫快取檔（*.cache），所以先把 test 90 張複製到暫存資料夾，
不動 data/；data/scenes/helmet.yaml 的 path 是相對路徑，也在暫存資料夾另寫一份。
"""
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

from detlib import SCENES, WEIGHTS

tmp = Path(tempfile.mkdtemp(prefix="yolo-val-"))
for sub in ("images", "labels"):
    shutil.copytree(SCENES / sub / "test", tmp / sub / "test")
yaml = tmp / "scenes.yaml"
yaml.write_text(f"path: {tmp}\ntrain: images/test\nval: images/test\nnames:\n  0: helmet\n  1: head\n")

m = YOLO(WEIGHTS).val(data=str(yaml), split="val", imgsz=320, batch=16, conf=0.001, iou=0.7,
                      plots=False, verbose=False, project=str(tmp), name="val")
print("== 4. model.val() 在 scenes test 90 張 ==")
print(f"全部類別：P {m.box.mp:.3f}  R {m.box.mr:.3f}  mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")
for i, name in m.names.items():
    print(f"{name:6s}：P {m.box.p[i]:.3f}  R {m.box.r[i]:.3f}  AP50 {m.box.ap50[i]:.3f}  AP50-95 {m.box.ap[i]:.3f}")

shutil.rmtree(tmp, ignore_errors=True)
