"""Training loop with per-epoch history and an optional progress callback."""
import torch
import torch.nn as nn
from .model import build_model, make_optimizer


def train(data, hidden=(64,), activation="ReLU", optimizer="Adam", lr=1e-3,
          epochs=30, batch_size=32, seed=42, on_epoch=None):
    torch.manual_seed(seed)
    model = build_model(hidden, activation)
    crit, opt = nn.CrossEntropyLoss(), make_optimizer(optimizer, model.parameters(), lr)
    X, y, Xt, yt = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    n = len(X)
    h = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for ep in range(epochs):
        model.train()
        perm, tot, hit = torch.randperm(n), 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(X[idx])
            loss = crit(out, y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            hit += (out.argmax(1) == y[idx]).sum().item()
        model.eval()
        with torch.no_grad():
            o = model(Xt)
            h["test_loss"].append(crit(o, yt).item())
            h["test_acc"].append((o.argmax(1) == yt).float().mean().item())
        h["train_loss"].append(tot / n)
        h["train_acc"].append(hit / n)
        if on_epoch:
            on_epoch(ep, h)
    return model, h
