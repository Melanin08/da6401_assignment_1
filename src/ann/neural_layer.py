"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np
from .activations import get_activation, get_activation_derivative


class NeuralLayer:
    def __init__(self, input_size, output_size, activation="relu", weight_init="xavier"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation

        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size)).astype(np.float64)
        elif weight_init == "random":
            self.W = (0.01 * np.random.randn(input_size, output_size)).astype(np.float64)
        elif weight_init == "zeros":
            self.W = np.zeros((input_size, output_size), dtype=np.float64)
        else:
            raise ValueError("weight_init must be one of: random, xavier, zeros")

        self.b = np.zeros((1, output_size), dtype=np.float64)

        self.activation = get_activation(activation)
        self.activation_derivative = get_activation_derivative(activation)

        self.input = None
        self.z = None
        self.a = None

        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.W) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, grad_output):
        if self.activation_name == "relu":
            grad_activation = grad_output * self.activation_derivative(self.z)
        else:
            grad_activation = grad_output * self.activation_derivative(self.a)

        self.grad_W = np.dot(self.input.T, grad_activation)
        self.grad_b = np.sum(grad_activation, axis=0, keepdims=True)

        grad_input = np.dot(grad_activation, self.W.T)
        return grad_input