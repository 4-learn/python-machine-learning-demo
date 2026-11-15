"""18｜PyTorch 的訓練迴圈：DataLoader 一批一批餵資料、CrossEntropyLoss 量錯多少、Adam 把參數往少錯的方向挪。"""
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
torch.set_num_threads(4)

print("== 1. TensorDataset：把特徵和答案綁成一筆一筆 ==")
x = torch.arange(10, dtype=torch.float32).reshape(10, 1)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
ds = TensorDataset(x, y)
print(f"共 {len(ds)} 筆；第 7 筆：{ds[7]}")

print("\n== 2. DataLoader：每次交出一批，每一輪先洗牌 ==")
loader = DataLoader(ds, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))
print(f"batch_size=4，10 筆分成 {len(loader)} 批")
for epoch in [1, 2]:
    batches = [xb.flatten().int().tolist() for xb, yb in loader]
    print(f"第 {epoch} 輪：" + "  ".join(str(b) for b in batches))

print("\n== 3. CrossEntropyLoss：猜得越有把握又越錯，罰越重 ==")
loss_fn = nn.CrossEntropyLoss()
rows = []
for scores, answer in [([-2.0, 2.0], 1), ([0.0, 0.5], 1), ([0.0, 0.0], 1), ([2.0, -2.0], 1), ([2.0, -2.0], 0)]:
    logits = torch.tensor([scores])
    prob = logits.softmax(1)[0, answer].item()
    loss = loss_fn(logits, torch.tensor([answer])).item()
    rows.append({"ok 分數": scores[0], "alert 分數": scores[1], "正確答案": answer,
                 "正確答案的機率": round(prob, 3), "loss": round(loss, 3)})
print(pd.DataFrame(rows).to_string(index=False))


def make_pictures(n, seed):
    """n 張 4×4 的小圖（16 個像素，0～1）：ok 的最上面一排比較亮（像戴著帽子），alert 沒有。"""
    g = torch.Generator().manual_seed(seed)
    pics = torch.rand(n, 4, 4, generator=g) * 0.6
    label = torch.randint(0, 2, (n,), generator=g)
    pics[label == 0, 0, :] += 0.4
    return pics.reshape(n, 16), label


x_train, y_train = make_pictures(200, seed=1)
x_valid, y_valid = make_pictures(100, seed=2)

print("\n== 4. 一步訓練：zero_grad → loss → backward → step ==")
model = nn.Linear(16, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
xb, yb = x_train[:32], y_train[:32]
w = model.weight
print(f"參數 {sum(p.numel() for p in model.parameters())} 個；一開始 weight[0, 0] = {w[0, 0].item():.4f}，梯度 {w.grad}")
optimizer.zero_grad()
loss = loss_fn(model(xb), yb)
loss.backward()
g = w.grad[0, 0].item()
print(f"backward 之後：梯度 = {g:.4f}（{'負的：這個權重調大' if g < 0 else '正的：這個權重調小'}，loss 會變小）")
optimizer.step()
print(f"step 之後：weight[0, 0] = {w[0, 0].item():.4f}")
with torch.no_grad():
    after = loss_fn(model(xb), yb).item()
print(f"同一批 32 張的 loss：{loss.item():.4f} → {after:.4f}")

print("\n== 5. 忘了 zero_grad：梯度會一直累加 ==")
model.zero_grad()
for i in [1, 2, 3]:
    loss_fn(model(xb), yb).backward()
    print(f"第 {i} 次 backward 之後，梯度 = {w.grad[0, 0].item():.4f}")


def train(lr, epochs=20):
    """完整的訓練迴圈，回傳每一輪的平均 loss（train）與 valid 的 loss。"""
    torch.manual_seed(0)
    model = nn.Linear(16, 2)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        with torch.no_grad():
            valid_loss = loss_fn(model(x_valid), y_valid).item()
        history.append({"epoch": epoch + 1, "train_loss": round(total / len(x_train), 3),
                        "valid_loss": round(valid_loss, 3)})
    return model, pd.DataFrame(history).set_index("epoch")


print("\n== 6. 訓練迴圈：一輪（epoch）＝ 200 張全部看過一次 ==")
model, history = train(lr=0.01)
n_batches = len(DataLoader(TensorDataset(x_train, y_train), batch_size=32))
print(f"batch_size=32：每一輪 {n_batches} 批，參數被調整 {n_batches} 次；20 輪共 {20 * n_batches} 次")
print(history.loc[[1, 2, 3, 5, 10, 15, 20]].to_string())
with torch.no_grad():
    pred = model(x_valid).argmax(1)
print(f"valid accuracy {(pred == y_valid).float().mean().item():.3f}")

print("\n== 7. lr：每一步跨多大 ==")
rows = {}
for lr in [0.0001, 0.01, 1.0, 50.0]:
    _, h = train(lr)
    rows[f"lr={lr}"] = h["train_loss"]
print(pd.DataFrame(rows).loc[[1, 2, 5, 10, 20]].to_string())
