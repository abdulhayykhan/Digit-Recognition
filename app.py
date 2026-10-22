"""Streamlit interface: predict digits, train your own ANN, browse experiments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

from src.data import load_data
from src.evaluate import confusion_figure, predict_labels, predict_proba
from src.model import load_model
from src.train import train

ROOT = Path(__file__).parent
st.set_page_config(page_title="Digit Recognition ANN", page_icon="🔢", layout="wide")


@st.cache_resource
def get_data():
    return load_data()


@st.cache_resource
def get_default_model():
    model, ck = load_model(ROOT / "models" / "digit_model.pt")
    return model, f"Default model ({ck['hidden']} hidden, {ck['activation']})"


@st.cache_data
def get_results():
    return json.loads((ROOT / "results" / "results.json").read_text())


def image_to_pixels(img, target_ink=0.2, thr=0.35):
    """Photo/drawing of a digit -> 8x8 array of 0-16 counts, built like the Digits dataset:
    tight square crop -> 32x32 bitmap -> on-pixel count per 4x4 block. Thin strokes are
    thickened (up to 3 times) until the ink density matches the training data."""
    g = np.asarray(ImageOps.grayscale(img), dtype=float)
    if g.mean() > 127:                       # dark ink on light paper -> invert
        g = 255 - g
    g = (g - g.min()) / max(g.max() - g.min(), 1)
    ys, xs = np.where(g > 0.3)
    if len(ys):                              # crop to the digit
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = max(g.shape)                      # square canvas (dataset digits fill the grid)
    canvas = np.zeros((side, side))
    r, c = (side - g.shape[0]) // 2, (side - g.shape[1]) // 2
    canvas[r:r + g.shape[0], c:c + g.shape[1]] = g
    big = Image.fromarray((canvas * 255).astype("uint8")).resize((32, 32), Image.LANCZOS)
    bits = np.asarray(big, dtype=float) / 255 > thr
    for _ in range(3):                       # match stroke thickness of the training data
        if bits.mean() >= target_ink:
            break
        big = big.filter(ImageFilter.MaxFilter(3))
        bits = np.asarray(big, dtype=float) / 255 > thr
    return bits.astype(float).reshape(8, 4, 8, 4).sum(axis=(1, 3))


def show_prediction(px, model, data, true=None):
    px = np.asarray(px, dtype=float).reshape(8, 8)   # test samples arrive as flat 64-vectors
    probs = predict_proba(model, px, data["mean"], data["scale"])
    pred = int(probs.argmax())
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.image(1 - np.kron(np.asarray(px) / 16, np.ones((24, 24))), clamp=True, caption="8×8 input")
    c2.metric("Prediction", pred)
    c2.metric("Confidence", f"{probs[pred]:.1%}")
    if true is not None:
        if pred == true:
            c2.success("Correct")
        else:
            c2.error(f"Wrong (true label: {true})")
    if probs[pred] < 0.6:
        c2.caption("Low confidence: treat this prediction with caution.")
    c3.bar_chart(pd.DataFrame({"probability": probs}, index=[str(i) for i in range(10)]))


data = get_data()
st.title("🔢 Handwritten Digit Recognition with an ANN")
st.caption("PyTorch feedforward network on the scikit-learn Digits dataset (8×8 images, 10 classes).")
tab_pred, tab_train, tab_exp, tab_about = st.tabs(["Predict", "Train your own", "Experiments", "About"])

# ---------------------------------------------------------------- Predict
with tab_pred:
    model, label = st.session_state.get("active", get_default_model())
    st.caption(f"Active model: {label}")
    mode = st.radio("Input", ["Test sample", "Edit pixels", "Upload image"], horizontal=True)
    if mode == "Test sample":
        i = st.slider("Test image index", 0, len(data["raw_test"]) - 1, 0)
        show_prediction(data["raw_test"][i], model, data, true=int(data["y_test"][i]))
    elif mode == "Edit pixels":
        base = st.number_input("Start from test sample (-1 = blank)", -1, len(data["raw_test"]) - 1, -1)
        grid = np.zeros((8, 8), int) if base < 0 else data["raw_test"][base].reshape(8, 8).astype(int)
        cfg = {str(c): st.column_config.NumberColumn(min_value=0, max_value=16, step=1, width="small")
               for c in range(8)}
        edited = st.data_editor(pd.DataFrame(grid, columns=[str(c) for c in range(8)]),
                                column_config=cfg, hide_index=True, key=f"grid_{base}")
        st.caption("Pixel values 0 (blank) to 16 (full ink). Edit cells to change the digit.")
        show_prediction(edited.to_numpy(float), model, data)
    else:
        up = st.file_uploader("Upload a picture of a single digit (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if up:
            show_prediction(image_to_pixels(Image.open(up)), model, data)
            st.caption("Uploads are out-of-distribution for this model: on 1,500 real MNIST test digits it reaches about 55%, versus 97% on its own 8×8 test set. Expect frequent mistakes.")

# ---------------------------------------------------------------- Train
with tab_train:
    with st.form("cfg"):
        c1, c2, c3 = st.columns(3)
        layers = c1.slider("Hidden layers", 1, 4, 1)
        neurons = c1.select_slider("Neurons per layer", [8, 16, 32, 64, 128, 256], 64)
        act = c2.selectbox("Activation", ["ReLU", "Sigmoid", "Tanh"])
        opt = c2.selectbox("Optimizer", ["Adam", "RMSProp", "Momentum", "SGD"])
        lr = c3.select_slider("Learning rate", [1e-4, 1e-3, 1e-2, 1e-1], 1e-3, format_func=lambda v: f"{v:g}")
        epochs = c3.slider("Epochs", 5, 100, 30, 5)
        go = st.form_submit_button("Train", type="primary")
    if go:
        bar, chart = st.progress(0.0), st.empty()

        def on_epoch(ep, h):
            bar.progress((ep + 1) / epochs)
            chart.line_chart(pd.DataFrame({"train loss": h["train_loss"], "test loss": h["test_loss"]}))

        hidden = (neurons,) * layers
        m, h = train(data, hidden=hidden, activation=act, optimizer=opt, lr=lr, epochs=epochs, on_epoch=on_epoch)
        bar.empty(); chart.empty()
        st.session_state["last"] = dict(model=m, h=h, label=f"{hidden} hidden, {act}, {opt}, lr={lr:g}")
    last = st.session_state.get("last")
    if last:
        h = last["h"]
        st.subheader(f"Result: {last['label']}")
        best = max(h["test_acc"])
        if best - h["test_acc"][-1] >= 0.05:
            st.warning(f"Training became unstable: test accuracy peaked at {best:.1%} (epoch "
                       f"{h['test_acc'].index(best) + 1}) then fell to {h['test_acc'][-1]:.1%}. "
                       "This usually means the learning rate is too high for this network; try a lower one.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Test accuracy", f"{h['test_acc'][-1]:.2%}")
        m2.metric("Train accuracy", f"{h['train_acc'][-1]:.2%}")
        m3.metric("Test loss", f"{h['test_loss'][-1]:.4f}")
        a, b, c = st.columns([1, 1, 1])
        a.line_chart(pd.DataFrame({"train": h["train_loss"], "test": h["test_loss"]}), y_label="loss")
        b.line_chart(pd.DataFrame({"train": h["train_acc"], "test": h["test_acc"]}), y_label="accuracy")
        c.pyplot(confusion_figure(data["y_test"].numpy(), predict_labels(last["model"], data["X_test"])))
        if st.button("Use this model in the Predict tab"):
            st.session_state["active"] = (last["model"], f"Your model ({last['label']})")
            st.success("Done. Open the Predict tab.")

# ---------------------------------------------------------------- Experiments
with tab_exp:
    res = get_results()
    names = {"architecture": "Layers & neurons", "learning_rate": "Learning rate",
             "activation": "Activation function", "optimizer": "Optimizer"}
    key = st.selectbox("Experiment", list(names), format_func=names.get)
    runs = res["experiments"][key]
    st.dataframe(pd.DataFrame([{"Configuration": k, "Train loss": r["final"]["train_loss"],
                                "Train acc": r["final"]["train_acc"], "Test loss": r["final"]["test_loss"],
                                "Test acc": r["final"]["test_acc"]} for k, r in runs.items()]),
                 hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.4f")
                                                 for c in ["Train loss", "Train acc", "Test loss", "Test acc"]})
    a, b = st.columns(2)
    a.line_chart(pd.DataFrame({k: r["history"]["train_loss"] for k, r in runs.items()}), y_label="train loss")
    b.line_chart(pd.DataFrame({k: r["history"]["test_acc"] for k, r in runs.items()}), y_label="test accuracy")
    st.caption("Single run per configuration (seed 42); differences of ~1 point are within noise.")
    st.subheader("Baseline")
    a, b = st.columns([3, 2])
    a.image(str(ROOT / "results" / "curves.png"))
    b.image(str(ROOT / "results" / "confusion_matrix.png"))

# ---------------------------------------------------------------- About
with tab_about:
    st.markdown("""
**Problem:** classify 8×8 handwritten digit images (0–9).
**Data:** scikit-learn Digits, 1,797 samples, stratified 80/20 split, standardized with training statistics.
**Baseline:** 64 → 64 (ReLU) → 10, Cross-Entropy loss, Adam (lr 0.001), batch 32, 30 epochs → **97.22% test accuracy**.

**Findings:** learning rate and optimizer matter most (lr 0.1 is unstable, plain SGD is slowest);
extra depth does not help on this small dataset; ReLU and Tanh converge faster than Sigmoid.

**Limitations:** single runs, small low-resolution dataset; photos of digits are less accurate than the test set.
""")