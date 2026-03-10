"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np
from .activations import softmax


def _ensure_one_hot(y_true, num_classes):
    y_true = np.asarray(y_true)

    if y_true.ndim == 2:
        return y_true.astype(np.float64)

    y_true = y_true.astype(np.int64).reshape(-1)
    one_hot = np.zeros((y_true.shape[0], num_classes), dtype=np.float64)
    one_hot[np.arange(y_true.shape[0]), y_true] = 1.0
    return one_hot


def cross_entropy_loss(logits, y_true):
    probs = softmax(logits)
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    y_true_oh = _ensure_one_hot(y_true, probs.shape[1])
    loss = -np.sum(y_true_oh * np.log(probs)) / y_true_oh.shape[0]
    return loss, probs


def cross_entropy_grad(probs, y_true):
    y_true_oh = _ensure_one_hot(y_true, probs.shape[1])
    batch_size = y_true_oh.shape[0]
    return (probs - y_true_oh) / batch_size


def mse_loss(predictions, y_true):
    y_true_oh = _ensure_one_hot(y_true, predictions.shape[1])
    return np.mean((predictions - y_true_oh) ** 2)


def mse_grad(predictions, y_true):
    y_true_oh = _ensure_one_hot(y_true, predictions.shape[1])
    return 2.0 * (predictions - y_true_oh) / predictions.size


def get_loss(name):
    name = (name or "cross_entropy").lower().strip()
    if name in ["cross_entropy", "crossentropy"]:
        return "cross_entropy"
    if name in ["mse", "mean_squared_error"]:
        return "mse"
    raise ValueError("loss must be one of: cross_entropy, mse")
