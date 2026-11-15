"""15｜joblib 與 hashlib：把訓練好的模型存成檔案，啟動時用 sha256 指紋確認是同一個檔案。"""
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

train = pd.DataFrame({
    "confidence": [0.95, 0.83, 0.81, 0.72, 0.66, 0.55, 0.49, 0.41, 0.35, 0.92],
    "duration_seconds": [80.0, 12.0, 65.0, 50.0, 9.0, 11.0, 70.0, 8.0, 6.0, 15.0],
    "label": [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
})
new = pd.DataFrame({"confidence": [0.90, 0.60, 0.40], "duration_seconds": [75.0, 10.0, 60.0]})

print("== 1. 訓練一個小模型 ==")
model = make_pipeline(StandardScaler(), LogisticRegression()).fit(train[["confidence", "duration_seconds"]],
                                                                  train["label"])
before = model.predict_proba(new)[:, 1].round(3)
print(f"新事件的分數：{before.tolist()}")

print("\n== 2. joblib.dump：存成檔案 ==")
path = OUT / "model.joblib"
joblib.dump(model, path)
print(f"{path.name}：{path.stat().st_size} bytes")

print("\n== 3. joblib.load：另一支程式讀回來，分數一樣 ==")
loaded = joblib.load(path)
after = loaded.predict_proba(new)[:, 1].round(3)
print(f"讀回來的分數：{after.tolist()}；和存檔前相同：{bool((before == after).all())}")
print(f"讀回來的是：{type(loaded).__name__}，步驟 {[name for name, _ in loaded.steps]}")

print("\n== 4. hashlib.sha256：檔案的指紋 ==")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"完整：{digest}（{len(digest)} 個字元）")
print(f"取前 12 碼：{digest[:12]}")
print(f"同一個檔案再算一次，相同：{hashlib.sha256(path.read_bytes()).hexdigest() == digest}")

print("\n== 5. 檔案只差 1 個 byte，指紋就整個不同 ==")
broken = OUT / "model-broken.joblib"
broken.write_bytes(path.read_bytes() + b"\0")
print(f"{broken.name}：{hashlib.sha256(broken.read_bytes()).hexdigest()[:12]}")
print(f"joblib.load 照樣讀得進來：{type(joblib.load(broken)).__name__}")

print("\n== 6. 設定檔記下指紋，啟動時比對 ==")
(OUT / "model.json").write_text(json.dumps({"high": 0.8, "low": 0.3, "sha256_12": digest[:12]}, indent=1))
config = json.loads((OUT / "model.json").read_text())
for candidate in [path, broken]:
    try:
        if hashlib.sha256(candidate.read_bytes()).hexdigest()[:12] != config["sha256_12"]:
            raise RuntimeError("模型檔和設定檔對不上，拒絕啟動")
        print(f"{candidate.name}：指紋相符，啟動")
    except RuntimeError as e:
        print(f"{candidate.name}：{e}")
