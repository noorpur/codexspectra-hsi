from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from codexspectra.models.deep import SpectralMLP
from codexspectra.utils.repro import device_name


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 1337,
    device: str = "auto",
):
    torch.manual_seed(seed)
    dev = torch.device(device_name(device))
    le = LabelEncoder().fit(np.concatenate([y_train, y_val]))
    yt = le.transform(y_train)
    yv = le.transform(y_val)
    model = SpectralMLP(x_train.shape[1], len(le.classes_)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(yt, dtype=torch.long))
    val_x = torch.tensor(x_val, dtype=torch.float32).to(dev)
    val_y = torch.tensor(yv, dtype=torch.long).to(dev)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    best = {"loss": float("inf"), "state": None}
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model.cpu(), le, best["loss"]
