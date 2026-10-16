"""
PhytoSense - leaf disease classifier (CNN trained on the PlantVillage dataset)
==============================================================================
A small convolutional neural network (5 conv blocks, ~0.5 M weights) trained on
16,354 PlantVillage leaf photos: 38 classes = 14 crops x (healthy + common diseases).

Runs with NumPy only (no TensorFlow/PyTorch needed on the laptop): the trained weights
are stored in models/leaf_cnn.npz with BatchNorm already folded into the convolutions.

Input: 96 x 96 RGB pixels (the browser crops the photo to a centred square and shrinks it).
Output: top-3 classes with probabilities.

Honest limits: it only knows the 14 PlantVillage crops (apple, blueberry, cherry, corn,
grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry, tomato).
It was trained on lab-style photos of single leaves on a plain background, so real
photos (other plants such as money plant, busy backgrounds) are less reliable. The AI
vision model is told this and only uses the CNN result when the plant matches.
"""

import os
import base64
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "leaf_cnn.npz")
SIZE = 96


def _pretty(cls):
    """'Tomato___Early_blight' -> ('Tomato', 'Early blight')"""
    crop, _, cond = cls.partition("___")
    crop = crop.replace("_", " ").replace(",", "").replace("(including sour)", "").replace("(maize)", "(maize)").strip()
    cond = cond.replace("_", " ").strip()
    return " ".join(crop.split()), " ".join(cond.split())


class LeafClassifier:
    def __init__(self, path=MODEL_PATH):
        d = np.load(path, allow_pickle=False)
        self.convs = []
        i = 0
        while f"conv{i}_w" in d:
            self.convs.append((d[f"conv{i}_w"].astype(np.float32), d[f"conv{i}_b"].astype(np.float32)))
            i += 1
        self.dense_w = d["dense_w"].astype(np.float32)
        self.dense_b = d["dense_b"].astype(np.float32)
        self.classes = [str(c) for c in d["classes"]]
        self.test_accuracy = float(d["test_accuracy"]) if "test_accuracy" in d else None
        self.crops = sorted({_pretty(c)[0] for c in self.classes})

    @staticmethod
    def _conv_relu_pool(x, w, b):
        # x: (H, W, Cin), w: (3, 3, Cin, Cout) -> same-padding conv + ReLU + 2x2 max-pool
        xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
        win = sliding_window_view(xp, (3, 3), axis=(0, 1))            # (H, W, Cin, 3, 3)
        y = np.einsum("hwcij,ijco->hwo", win, w, optimize=True) + b
        y = np.maximum(y, 0.0)
        h, ww = (y.shape[0] // 2) * 2, (y.shape[1] // 2) * 2
        y = y[:h, :ww].reshape(h // 2, 2, ww // 2, 2, -1).max(axis=(1, 3))
        return y

    def predict_pixels(self, rgb, top=3):
        """rgb: uint8 array (96, 96, 3). Returns top classes with probabilities."""
        x = rgb.astype(np.float32) / 255.0
        for w, b in self.convs:
            x = self._conv_relu_pool(x, w, b)
        feat = x.mean(axis=(0, 1))
        logits = feat @ self.dense_w + self.dense_b
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-p)[:top]
        out = []
        for k in order:
            crop, cond = _pretty(self.classes[k])
            out.append({"class": self.classes[k], "crop": crop, "condition": cond,
                        "healthy": cond.lower() == "healthy", "probability": round(float(p[k]), 3)})
        return out

    def predict_b64(self, pixels_b64, top=3):
        """pixels_b64: base64 of raw RGB bytes (96*96*3) sent by the browser."""
        raw = base64.b64decode(pixels_b64)
        if len(raw) != SIZE * SIZE * 3:
            raise ValueError("expected 96x96 RGB pixels")
        return self.predict_pixels(np.frombuffer(raw, dtype=np.uint8).reshape(SIZE, SIZE, 3), top)


def load_classifier():
    try:
        clf = LeafClassifier()
        acc = f", test accuracy {clf.test_accuracy:.1%}" if clf.test_accuracy else ""
        print(f"[PhytoSense] Leaf CNN loaded: {len(clf.classes)} classes, {len(clf.crops)} crops{acc}")
        return clf
    except FileNotFoundError:
        print("[PhytoSense] Leaf CNN not found (models/leaf_cnn.npz) - photo check uses the AI vision model only")
    except Exception as e:
        print(f"[PhytoSense] Leaf CNN could not be loaded: {e}")
    return None
