"""Train the baseline model, save it, and run all experiments.

    python train.py
Outputs: models/digit_model.pt, results/results.json, results/*.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import load_data
from src.evaluate import confusion_figure, predict_labels
from src.model import save_model
from src.train import train

BASE = dict(hidden=(64,), activation="ReLU", optimizer="Adam", lr=1e-3, epochs=30, batch_size=32)
EXPERIMENTS = {
    "architecture": {f"{len(h)}x{h[0]}": dict(hidden=h) for h in
                     [(16,), (64,), (128,), (64, 64), (64, 64, 64)]},
    "learning_rate": {f"lr={v}": dict(lr=v) for v in [1e-4, 1e-3, 1e-2, 1e-1]},
    "activation": {a: dict(activation=a) for a in ["ReLU", "Sigmoid", "Tanh"]},
    "optimizer": {"SGD": dict(optimizer="SGD", lr=0.01), "Momentum": dict(optimizer="Momentum", lr=0.01),
                  "RMSProp": dict(optimizer="RMSProp"), "Adam": dict(optimizer="Adam")},
}


def record(data, **over):
    cfg = {**BASE, **over}
    model, h = train(data, **cfg)
    final = {k: v[-1] for k, v in h.items()}
    print(f"  {over or 'baseline'}: test acc {final['test_acc']:.4f}")
    return model, cfg, {"config": {**cfg, "hidden": list(cfg["hidden"])}, "history": h, "final": final}


def main():
    data = load_data()
    print("Baseline")
    model, cfg, base = record(data)
    save_model(model, cfg["hidden"], cfg["activation"], "models/digit_model.pt")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    h = base["history"]
    for a, (k, t) in zip(ax, [("loss", "Loss"), ("acc", "Accuracy")]):
        a.plot(h[f"train_{k}"], label="Train"); a.plot(h[f"test_{k}"], label="Test")
        a.set_title(t); a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("results/curves.png", dpi=130); plt.close(fig)
    fig = confusion_figure(data["y_test"].numpy(), predict_labels(model, data["X_test"]))
    fig.savefig("results/confusion_matrix.png", dpi=130); plt.close(fig)

    out = {"baseline": base, "experiments": {}}
    for name, runs in EXPERIMENTS.items():
        print(name)
        out["experiments"][name] = {label: record(data, **over)[2] for label, over in runs.items()}
    json.dump(out, open("results/results.json", "w"))


if __name__ == "__main__":
    main()
