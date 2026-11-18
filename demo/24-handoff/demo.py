"""24｜hashlib：替考卷算 sha256 指紋，寫成 manifest.json，收到的人重算一次就知道有沒有被動過。"""
import hashlib
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# 一份 6 張照片的小考卷：照片、拍攝資訊、事後答案
exam = pd.DataFrame({
    "image": ["test_000.jpg", "test_001.jpg", "test_006.jpg", "test_014.jpg", "test_023.jpg", "test_026.jpg"],
    "location": ["rest_01", "dock_01", "rest_01", "dock_01", "aisle_01", "aisle_01"],
    "duration_seconds": [149.7, 77.8, 28.2, 13.1, 34.6, 7.2],
    "should_alert": [0, 1, 0, 0, 1, 0],
})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


print("== 1. 考卷存成 CSV，算指紋 ==")
path = OUT / "eval-set.csv"
exam.to_csv(path, index=False, lineterminator="\n")
first = sha256(path)
print(path.read_text(), end="")
print(f"sha256：{first}")

print("\n== 2. 同樣的內容再寫一次，指紋一樣 ==")
exam.to_csv(path, index=False, lineterminator="\n")
print(f"再寫一次：{sha256(path) == first}")

print("\n== 3. 改一個答案（1 個 byte），指紋整個不同 ==")
data = bytearray(path.read_bytes())
data[-2] = ord("1")                 # 倒數第 2 個 byte＝最後一張照片的 should_alert：0 → 1
tampered = OUT / "eval-set-tampered.csv"
tampered.write_bytes(bytes(data))
second = sha256(tampered)
print(f"原本：{first[:16]}…")
print(f"改後：{second[:16]}…")
print(f"64 個字元裡，相同位置一樣的有 {sum(a == b for a, b in zip(first, second))} 個")

print("\n== 4. json.dumps：把指紋寫成 manifest.json ==")
manifest = {"line": "factory", "images": len(exam), "should_alert": int(exam["should_alert"].sum()),
            "eval_set_sha256": first}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
print((OUT / "manifest.json").read_text(encoding="utf-8"))

print("\n== 5. 收到交接包的人：重算一次，和 manifest 比 ==")
received = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
for candidate in [path, tampered]:
    ok = sha256(candidate) == received["eval_set_sha256"]
    print(f"{candidate.name:<22} {'相符，可以用' if ok else '對不上，不要用'}")
