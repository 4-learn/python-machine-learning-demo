"""17｜共用小工具：讀頭部小圖、建立「凍結骨幹」的 MobileNetV3-Small、訓練最後一層。

只有 17 節的腳本會 import 它（從 repo 根目錄執行 python 17-finetune/xxx.py 時，
Python 會把 17-finetune/ 放進搜尋路徑）。
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from common import DATA

torch.set_num_threads(4)
WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1
CROPS = DATA / "crops"
NORMALIZE = T.Normalize(WEIGHTS.transforms().mean, WEIGHTS.transforms().std)


def load_index() -> pd.DataFrame:
    """index.csv：file, camera, label, kind。y = 1 表示沒戴安全帽（要通知的那一類）。"""
    idx = pd.read_csv(CROPS / "index.csv")
    idx["y"] = (idx["label"] == "no_helmet").astype(int)
    return idx


def load_images(files) -> torch.Tensor:
    """讀成 (N, 3, 96, 96)、數值 0～1 的 tensor。"""
    return torch.stack([T.functional.to_tensor(Image.open(CROPS / f).convert("RGB")) for f in files])


def build_model() -> nn.Module:
    """預訓練 MobileNetV3-Small：凍結全部參數，最後一層換成 2 類（helmet / no_helmet）。"""
    model = mobilenet_v3_small(weights=WEIGHTS)
    for p in model.parameters():
        p.requires_grad = False                  # 凍結：這些參數不再更新
    model.classifier[3] = nn.Linear(1024, 2)     # 新的最後一層：預設 requires_grad=True
    return model


def backbone_features(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """骨幹＋分類頭前兩層的輸出（每張圖 1024 個數字）。凍結的部分輸出不會變，所以只要算一次。"""
    model.eval()
    with torch.no_grad():
        return torch.cat([model.classifier[:3](model.avgpool(model.features(NORMALIZE(X[i:i + 64]))).flatten(1))
                          for i in range(0, len(X), 64)])


def train_head(head: nn.Linear, F: torch.Tensor, y, epochs=30, seed=0) -> list[float]:
    """只訓練最後一層。回傳每個 epoch 的平均 loss。"""
    torch.manual_seed(seed)
    head.reset_parameters()
    y = torch.tensor(np.array(y), dtype=torch.long)
    loader = DataLoader(TensorDataset(F, y), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(head.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        history.append(total / len(y))
    return history


def prob_no_helmet(head: nn.Linear, F: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return head(F).softmax(dim=1)[:, 1].numpy()


def darken(X: torch.Tensor, seed=0) -> torch.Tensor:
    """模擬夜間：整張變暗（0.35～0.7 倍）再加一點雜訊。"""
    g = torch.Generator().manual_seed(seed)
    scale = torch.empty(len(X), 1, 1, 1).uniform_(0.35, 0.7, generator=g)
    return (X * scale + torch.randn(X.shape, generator=g) * 0.03).clamp(0, 1)
