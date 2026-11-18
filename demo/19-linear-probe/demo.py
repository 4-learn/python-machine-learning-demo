"""19｜scikit-learn LogisticRegression：linear probe。預訓練網路負責把小圖變成數字，sklearn 的 Pipeline 負責學判斷。"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import get_model, get_model_weights

torch.set_num_threads(4)
CROPS = Path("story/data/factory/crops")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
BACKBONE = "resnet18"  # 18 節工廠線在 valid 上選出來的骨幹

print("== 1. 骨幹只負責看：每張小圖變成 512 個數字 ==")
index = pd.read_csv(CROPS / "index.csv")
weights = get_model_weights("resnet18")["IMAGENET1K_V1"]
net = get_model("resnet18", weights=weights).eval()
net.fc = torch.nn.Identity()
pre = weights.transforms()
with torch.no_grad():
    x = torch.cat([net(torch.stack([pre(Image.open(CROPS / f).convert("RGB")) for f in index["file"][i:i + 32]]))
                   for i in range(0, len(index), 32)]).numpy()
y = (index["label"] == "alert").astype(int).to_numpy()
print(f"x：{type(x).__name__} {x.shape}，y：{y.shape}，alert 占 {y.mean():.3f}")
part = np.select([index["camera"] == "camera_04", index["camera"] == "camera_03"], ["test", "valid"], "train")
tr, va, te = part == "train", part == "valid", part == "test"
print(f"train {tr.sum()} 張（camera_01、02）、valid {va.sum()} 張（camera_03）、test {te.sum()} 張（夜間 camera_04）")


def report(model, mask):
    pred = model.predict(x[mask])
    return {"accuracy": round(float((pred == y[mask]).mean()), 3),
            "precision": round(precision_score(y[mask], pred, zero_division=0), 3),
            "recall": round(recall_score(y[mask], pred), 3)}


print("\n== 2. Pipeline(StandardScaler, LogisticRegression) vs KNN（k=1）==")
probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(x[tr], y[tr])
knn = KNeighborsClassifier(n_neighbors=1).fit(x[tr], y[tr])
rows = [{"模型": name, "資料": s, **report(m, mask)}
        for name, m in [("KNN（k=1）", knn), ("邏輯迴歸", probe)] for s, mask in [("valid", va), ("test", te)]]
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 3. predict_proba：夜間攝影機上，邏輯迴歸叫錯的那幾張 ==")
night = index[te].assign(alert_機率=probe.predict_proba(x[te])[:, 1].round(3), 預測=probe.predict(x[te]))
wrong = night[night["預測"] != (night["label"] == "alert")]
print(wrong[["file", "kind", "label", "alert_機率"]].sort_values("alert_機率").to_string(index=False))
false_alarm = int((wrong["label"] == "ok").sum())
print(f"夜間 {te.sum()} 張錯 {len(wrong)} 張：誤報（ok 說成 alert）{false_alarm} 張、漏報 {len(wrong) - false_alarm} 張")

print("\n== 4. joblib：骨幹名稱和 Pipeline 存在一起 ==")
joblib.dump({"backbone": BACKBONE, "probe": probe}, OUT / "probe.joblib")
saved = joblib.load(OUT / "probe.joblib")
same = (saved["probe"].predict(x[te]) == probe.predict(x[te])).all()
print(f"probe.joblib：{(OUT / 'probe.joblib').stat().st_size:,} bytes；內容 {list(saved)}，骨幹 {saved['backbone']}")
print(f"讀回來的 Pipeline，test 的答案和原本相同：{bool(same)}")
