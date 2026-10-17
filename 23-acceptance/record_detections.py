"""23｜真實模式跑一次 YOLO，把每張圖的 head 框存成 detections.json（Mock 模式讀它），並確認兩者一致。

    python 23-acceptance/record_detections.py      # CPU 約 10 秒
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from system import MOCK_DETECTIONS, SafetyDetector, load_frames  # noqa: E402

frames = load_frames()
live = SafetyDetector()
saved = {Path(image).name: live.detect(image) for image, *_ in frames}
MOCK_DETECTIONS.write_text(json.dumps(saved, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

mock = SafetyDetector(mock=True)
print("== 真實模式 vs Mock ==")
print(f"{len(saved)} 張圖，head 框 {sum(map(len, saved.values()))} 個（門檻 {live.min_confidence}）")
print("Mock 讀回來完全相同：", all(mock.detect(p) == live.detect(p) for p, *_ in frames))
