"""16｜在表格事件上手寫 PyTorch 訓練迴圈，和 08 節的梯度提升比較。"""
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import FEATURES, TARGET, current_rule, fit_valid_split, load_events, make_preprocessor, score_row, time_split

torch.set_num_threads(4)
torch.manual_seed(0)

train, _ = time_split(load_events())
fit, valid = fit_valid_split(train)

# 前處理沿用 06 節：只在 fit 上 fit，valid 只 transform
pre = make_preprocessor().fit(fit[FEATURES])


class EventDataset(Dataset):
    """把事件表包成 PyTorch 認得的資料集：知道有幾筆、能取出第 i 筆。"""

    def __init__(self, df):
        self.X = torch.tensor(pre.transform(df[FEATURES]), dtype=torch.float32)
        self.y = torch.tensor(df[TARGET].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


fit_ds, valid_ds = EventDataset(fit), EventDataset(valid)
loader = DataLoader(fit_ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))

print("== 1. Dataset 與 DataLoader ==")
x0, y0 = fit_ds[0]
print(f"fit {len(fit_ds)} 筆、valid {len(valid_ds)} 筆；一筆 = {x0.shape[0]} 個數字（7 數值 + 12 個 one-hot）")
xb, yb = next(iter(loader))
print(f"一個 batch：X {tuple(xb.shape)}、y {tuple(yb.shape)}；一個 epoch = {len(loader)} 個 batch")

print("\n== 2. loss：猜得越有把握又越錯，罰越重 ==")
loss_fn = nn.BCELoss()
rows = []
for truth in [1.0, 0.0]:
    for p in [0.9, 0.6, 0.1]:
        rows.append({"真正答案": int(truth), "模型給的機率": p,
                     "loss": round(loss_fn(torch.tensor([p]), torch.tensor([truth])).item(), 3)})
print(pd.DataFrame(rows).to_string(index=False))

model = nn.Sequential(nn.Linear(19, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
loss_fn = nn.BCEWithLogitsLoss()      # 輸出不用自己做 sigmoid，數值較穩
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n== 3. 一步訓練：梯度告訴每個參數往哪邊調 ==")
w = model[0].weight
before = loss_fn(model(xb).squeeze(1), yb).item()
optimizer.zero_grad()
loss = loss_fn(model(xb).squeeze(1), yb)
loss.backward()                       # 算出每個參數的梯度
print(f"參數 {sum(p.numel() for p in model.parameters())} 個；第一個權重 {w[0, 0].item():+.4f}，梯度 {w.grad[0, 0].item():+.4f}")
optimizer.step()                      # 每個參數往「讓 loss 變小」的方向挪一小步
print(f"挪一步之後權重 {w[0, 0].item():+.4f}；同一個 batch 的 loss {before:.4f} → {loss_fn(model(xb).squeeze(1), yb).item():.4f}")


def mean_loss(ds):
    model.eval()
    with torch.no_grad():
        return loss_fn(model(ds.X).squeeze(1), ds.y).item()


print("\n== 4. 訓練迴圈：fit loss 一路降，valid loss 先降後升 ==")
history, best = [], {"loss": float("inf")}
for epoch in range(1, 61):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb).squeeze(1), yb)
        loss.backward()
        optimizer.step()
    fit_loss, valid_loss = mean_loss(fit_ds), mean_loss(valid_ds)
    history.append({"epoch": epoch, "fit_loss": round(fit_loss, 3), "valid_loss": round(valid_loss, 3)})
    if valid_loss < best["loss"]:
        best = {"loss": valid_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
hist = pd.DataFrame(history).set_index("epoch")
print(hist.loc[[1, 2, 5, 10, 15, 20, 30, 40, 50, 60]].to_string())
print(f"valid loss 最低在第 {best['epoch']} 個 epoch（{best['loss']:.3f}），之後模型開始背 fit 資料")

print("\n== 5. 成績表（valid，門檻 0.5）==")
model.load_state_dict(best["state"])
model.eval()
with torch.no_grad():
    mlp_pred = (torch.sigmoid(model(valid_ds.X).squeeze(1)) >= 0.5).int().numpy()
y = valid[TARGET]
lr = make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000)).fit(fit[FEATURES], fit[TARGET])
hgb = make_pipeline(make_preprocessor(), HistGradientBoostingClassifier(random_state=0)).fit(fit[FEATURES], fit[TARGET])
table = pd.DataFrame([
    score_row("多數類", y, np.zeros(len(y), dtype=int)),
    score_row("現行規則 conf>=0.7", y, current_rule(valid)),
    score_row("邏輯迴歸", y, lr.predict(valid[FEATURES])),
    score_row("梯度提升", y, hgb.predict(valid[FEATURES])),
    score_row(f"MLP（第 {best['epoch']} epoch）", y, mlp_pred),
])
print(table.to_string(index=False))

print("\n== 6. 換 5 個 seed 重訓（各 15 epoch）：MLP 的分數會晃 ==")
from sklearn.metrics import average_precision_score  # noqa: E402


def train_once(seed, epochs=15):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(19, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    dl = DataLoader(fit_ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(net(xb).squeeze(1), yb).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(valid_ds.X).squeeze(1)).numpy()


rows = []
for seed in range(5):
    prob = train_once(seed)
    r = score_row(f"MLP seed={seed}", y, (prob >= 0.5).astype(int))
    r["AP"] = round(average_precision_score(y, prob), 3)
    rows.append(r)
r = score_row("梯度提升", y, hgb.predict(valid[FEATURES]))
r["AP"] = round(average_precision_score(y, hgb.predict_proba(valid[FEATURES])[:, 1]), 3)
rows.append(r)
seeds = pd.DataFrame(rows)
print(seeds.to_string(index=False))
mlp = seeds.iloc[:5]
print(f"MLP recall {mlp.recall.mean():.3f} ± {mlp.recall.std():.3f}；AP {mlp.AP.mean():.3f} ± {mlp.AP.std():.3f}")
