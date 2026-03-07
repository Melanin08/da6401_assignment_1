"""
Optimization Algorithms
Implements: SGD, Momentum, nag, rmsprop
"""

import numpy as np


class SGDOptimizer:
    def __init__(self, learning_rate=0.001, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self, model):
        for layer in model.layers:
            layer.W -= self.learning_rate * (layer.grad_W + self.weight_decay * layer.W)
            layer.b -= self.learning_rate * layer.grad_b

        model.output_W -= self.learning_rate * (model.output_grad_W + self.weight_decay * model.output_W)
        model.output_b -= self.learning_rate * model.output_grad_b


class MomentumOptimizer:
    def __init__(self, learning_rate=0.001, beta=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = None

    def _initialize(self, model):
        self.velocities = []
        for layer in model.layers:
            self.velocities.append({
                "W": np.zeros_like(layer.W),
                "b": np.zeros_like(layer.b)
            })
        self.velocities.append({
            "W": np.zeros_like(model.output_W),
            "b": np.zeros_like(model.output_b)
        })

    def step(self, model):
        if self.velocities is None:
            self._initialize(model)

        for i, layer in enumerate(model.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.velocities[i]["W"] = self.beta * self.velocities[i]["W"] - self.learning_rate * grad_W
            self.velocities[i]["b"] = self.beta * self.velocities[i]["b"] - self.learning_rate * grad_b

            layer.W += self.velocities[i]["W"]
            layer.b += self.velocities[i]["b"]

        last = len(model.layers)
        grad_W = model.output_grad_W + self.weight_decay * model.output_W
        grad_b = model.output_grad_b

        self.velocities[last]["W"] = self.beta * self.velocities[last]["W"] - self.learning_rate * grad_W
        self.velocities[last]["b"] = self.beta * self.velocities[last]["b"] - self.learning_rate * grad_b

        model.output_W += self.velocities[last]["W"]
        model.output_b += self.velocities[last]["b"]


class NAGOptimizer:
    def __init__(self, learning_rate=0.001, beta=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = None

    def _initialize(self, model):
        self.velocities = []
        for layer in model.layers:
            self.velocities.append({
                "W": np.zeros_like(layer.W),
                "b": np.zeros_like(layer.b)
            })
        self.velocities.append({
            "W": np.zeros_like(model.output_W),
            "b": np.zeros_like(model.output_b)
        })

    def step(self, model):
        if self.velocities is None:
            self._initialize(model)

        for i, layer in enumerate(model.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            v_prev_W = self.velocities[i]["W"].copy()
            v_prev_b = self.velocities[i]["b"].copy()

            self.velocities[i]["W"] = self.beta * self.velocities[i]["W"] - self.learning_rate * grad_W
            self.velocities[i]["b"] = self.beta * self.velocities[i]["b"] - self.learning_rate * grad_b

            layer.W += -self.beta * v_prev_W + (1.0 + self.beta) * self.velocities[i]["W"]
            layer.b += -self.beta * v_prev_b + (1.0 + self.beta) * self.velocities[i]["b"]

        last = len(model.layers)
        grad_W = model.output_grad_W + self.weight_decay * model.output_W
        grad_b = model.output_grad_b

        v_prev_W = self.velocities[last]["W"].copy()
        v_prev_b = self.velocities[last]["b"].copy()

        self.velocities[last]["W"] = self.beta * self.velocities[last]["W"] - self.learning_rate * grad_W
        self.velocities[last]["b"] = self.beta * self.velocities[last]["b"] - self.learning_rate * grad_b

        model.output_W += -self.beta * v_prev_W + (1.0 + self.beta) * self.velocities[last]["W"]
        model.output_b += -self.beta * v_prev_b + (1.0 + self.beta) * self.velocities[last]["b"]


class RMSPropOptimizer:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = None

    def _initialize(self, model):
        self.cache = []
        for layer in model.layers:
            self.cache.append({
                "W": np.zeros_like(layer.W),
                "b": np.zeros_like(layer.b)
            })
        self.cache.append({
            "W": np.zeros_like(model.output_W),
            "b": np.zeros_like(model.output_b)
        })

    def step(self, model):
        if self.cache is None:
            self._initialize(model)

        for i, layer in enumerate(model.layers):
            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.cache[i]["W"] = self.beta * self.cache[i]["W"] + (1.0 - self.beta) * (grad_W ** 2)
            self.cache[i]["b"] = self.beta * self.cache[i]["b"] + (1.0 - self.beta) * (grad_b ** 2)

            layer.W -= self.learning_rate * grad_W / (np.sqrt(self.cache[i]["W"]) + self.epsilon)
            layer.b -= self.learning_rate * grad_b / (np.sqrt(self.cache[i]["b"]) + self.epsilon)

        last = len(model.layers)
        grad_W = model.output_grad_W + self.weight_decay * model.output_W
        grad_b = model.output_grad_b

        self.cache[last]["W"] = self.beta * self.cache[last]["W"] + (1.0 - self.beta) * (grad_W ** 2)
        self.cache[last]["b"] = self.beta * self.cache[last]["b"] + (1.0 - self.beta) * (grad_b ** 2)

        model.output_W -= self.learning_rate * grad_W / (np.sqrt(self.cache[last]["W"]) + self.epsilon)
        model.output_b -= self.learning_rate * grad_b / (np.sqrt(self.cache[last]["b"]) + self.epsilon)


def get_optimizer(name, learning_rate=0.001, weight_decay=0.0):
    name = name.lower().strip()

    if name == "sgd":
        return SGDOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "momentum":
        return MomentumOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "nag":
        return NAGOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "rmsprop":
        return RMSPropOptimizer(learning_rate=learning_rate, weight_decay=weight_decay)

    raise ValueError("optimizer must be one of: sgd, momentum, nag, rmsprop")