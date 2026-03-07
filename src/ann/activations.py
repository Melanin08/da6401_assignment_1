"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(output):
    return output * (1.0 - output)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(output):
    return 1.0 - np.square(output)


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    return (x > 0).astype(np.float64)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def get_activation(name):
    name = name.lower().strip()
    if name == "sigmoid":
        return sigmoid
    if name == "tanh":
        return tanh
    if name == "relu":
        return relu
    raise ValueError("activation must be one of: sigmoid, tanh, relu")


def get_activation_derivative(name):
    name = name.lower().strip()
    if name == "sigmoid":
        return sigmoid_derivative
    if name == "tanh":
        return tanh_derivative
    if name == "relu":
        return relu_derivative
    raise ValueError("activation must be one of: sigmoid, tanh, relu")