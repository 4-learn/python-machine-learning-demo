"""17｜微調 MobileNetV3-Small 分辨有沒有戴安全帽：依攝影機分組切割，cam5（夜間）留作測試。"""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import score_row
from crops import backbone_features, build_model, load_images, load_index, prob_no_helmet, train_head

idx = load_index()
X = load_images(idx.file)

print("== 1. 資料：6 支攝影機、660 張 96×96 頭部小圖 ==")
print(pd.crosstab(idx.camera, idx.label).to_string())

# 依攝影機分組：cam1–4 訓練、cam6 驗證、cam5（夜間）測試
part = idx.camera.map(lambda c: "test" if c == "cam5" else "valid" if c == "cam6" else "train")
tr, va, te = (np.array(part == p) for p in ["train", "valid", "test"])
print(f"train cam1–4 {tr.sum()} 張｜valid cam6 {va.sum()} 張｜test cam5（夜間）{te.sum()} 張")

print("\n== 2. 凍結骨幹，只留最後一層可以訓練 ==")
model = build_model()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"全部參數 {total:,}；要訓練的 {trainable:,}（{trainable / total:.1%}）")
print("新的最後一層：", model.classifier[3])

print("\n== 3. 訓練最後一層（30 epoch）==")
t0 = time.time()
F = backbone_features(model, X)          # 660 張各算一次骨幹
history = train_head(model.classifier[3], F[tr], idx.y[tr])
print(f"特徵 {tuple(F.shape)}；訓練時間 {time.time() - t0:.1f} 秒")
print("epoch  1  5  10  20  30 的 loss：", " ".join(f"{history[e - 1]:.3f}" for e in [1, 5, 10, 20, 30]))

print("\n== 4. 成績表：valid（cam6，白天）vs test（cam5，夜間）==")
y = idx.y.to_numpy()
head = model.classifier[3]
pixels = X[:, :, ::4, ::4].flatten(1).numpy()        # 不用預訓練：24×24 像素直接丟邏輯迴歸
lr = LogisticRegression(max_iter=3000).fit(pixels[tr], y[tr])
rows = []
for name, m in [("valid cam6", va), ("test cam5", te)]:
    rows += [
        score_row(f"{name}｜全猜有戴", y[m], np.zeros(m.sum(), dtype=int)),
        score_row(f"{name}｜像素+邏輯迴歸", y[m], lr.predict(pixels[m])),
        score_row(f"{name}｜微調 MobileNet", y[m], (prob_no_helmet(head, F[m]) >= 0.5).astype(int)),
    ]
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 5. 夜間的錯誤地圖（test cam5）==")
pred = (prob_no_helmet(head, F[te]) >= 0.5).astype(int)
names = {0: "有戴", 1: "沒戴"}
cm = pd.crosstab(pd.Series(y[te]).map(names).rename("真實"), pd.Series(pred).map(names).rename("預測"))
print(cm.to_string())
kind = pd.crosstab(idx.kind[te].rename("圖上畫的是"), pd.Series(pred, index=idx.index[te]).map(names).rename("預測"))
print(kind.to_string())

print("\n== 6. 門檻：0.5 不是唯一選擇 ==")
rows = []
for t in [0.3, 0.5, 0.7, 0.9]:
    pv, pt = prob_no_helmet(head, F[va]) >= t, prob_no_helmet(head, F[te]) >= t
    rows.append({"門檻": t, "valid recall": score_row("", y[va], pv)["recall"], "valid 誤報": int((pv & (y[va] == 0)).sum()),
                 "test recall": score_row("", y[te], pt)["recall"], "test 誤報": int((pt & (y[te] == 0)).sum())})
print(pd.DataFrame(rows).to_string(index=False))

print("\n== 7. 對照：如果隨機切割（同一支攝影機的照片跨兩邊）==")
i_tr, i_te = train_test_split(np.arange(len(idx)), test_size=110, random_state=0, stratify=y)
train_head(head, F[i_tr], y[i_tr])
print(pd.DataFrame([score_row("隨機切 110 張", y[i_te], (prob_no_helmet(head, F[i_te]) >= 0.5).astype(int))]).to_string(index=False))
