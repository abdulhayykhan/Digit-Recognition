"""ANN architecture, optimizer factory and save/load helpers."""
import torch
import torch.nn as nn
import torch.optim as optim

ACTIVATIONS = {"ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh}


def build_model(hidden=(64,), activation="ReLU", in_dim=64, out_dim=10):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), ACTIVATIONS[activation]()]
        d = h
    layers.append(nn.Linear(d, out_dim))  # logits; softmax is inside CrossEntropyLoss
    return nn.Sequential(*layers)


def make_optimizer(name, params, lr):
    if name == "SGD":
        return optim.SGD(params, lr=lr)
    if name == "Momentum":
        return optim.SGD(params, lr=lr, momentum=0.9)
    if name == "RMSProp":
        return optim.RMSprop(params, lr=lr)
    return optim.Adam(params, lr=lr)


def save_model(model, hidden, activation, path):
    torch.save({"state_dict": model.state_dict(), "hidden": list(hidden),
                "activation": activation}, path)


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    model = build_model(tuple(ck["hidden"]), ck["activation"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck
