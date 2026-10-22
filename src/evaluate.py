"""Prediction and evaluation helpers."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def predict_proba(model, pixels, mean, scale):
    """pixels: 8x8 (or 64) array with values 0-16 -> class probabilities."""
    x = (np.asarray(pixels, dtype=float).reshape(1, -1) - mean) / scale
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(x, dtype=torch.float32)), 1)[0].numpy()


def predict_labels(model, X):
    with torch.no_grad():
        return model(X).argmax(1).numpy()


def confusion_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    fig.colorbar(im); fig.tight_layout()
    return fig
