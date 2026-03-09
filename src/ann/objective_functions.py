"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np
from .activations import softmax


def cross_entropy_loss(logits, y_true):
    probs = softmax(logits)
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    loss = -np.sum(y_true * np.log(probs)) / y_true.shape[0]
    return loss, probs


def cross_entropy_grad(probs, y_true):
    batch_size = y_true.shape[0]
    return (probs - y_true) 


def mse_loss(predictions, y_true):
    return np.mean((predictions - y_true) ** 2)


def mse_grad(predictions, y_true):
    # because mse_loss uses np.mean over all elements
    return 2.0 * (predictions - y_true) / predictions.size


def get_loss(name):
    name = name.lower().strip()
    if name in ["cross_entropy", "crossentropy"]:
        return "cross_entropy"
    if name in ["mse", "mean_squared_error"]:
        return "mse"
    raise ValueError("loss must be one of: cross_entropy, mse")