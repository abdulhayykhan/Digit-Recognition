"""Load and preprocess the scikit-learn Digits dataset (8x8 images, 10 classes)."""
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, seed=42):
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    mean, scale = X_tr.mean(0), X_tr.std(0)
    scale[scale == 0] = 1.0  # constant pixels
    f = lambda a: torch.tensor((a - mean) / scale, dtype=torch.float32)
    return dict(X_train=f(X_tr), X_test=f(X_te),
                y_train=torch.tensor(y_tr), y_test=torch.tensor(y_te),
                raw_test=X_te, mean=mean, scale=scale)
