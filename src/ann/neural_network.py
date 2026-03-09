"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import json
import numpy as np

from .neural_layer import NeuralLayer
from .activations import softmax
from .objective_functions import (
    get_loss,
    cross_entropy_loss,
    cross_entropy_grad,
    mse_loss,
    mse_grad,
)
from .optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.dataset = getattr(cli_args, "dataset", "mnist")
        self.input_size = 784
        self.output_size = 10

        if hasattr(cli_args, "hidden_size"):
            self.hidden_layers = list(cli_args.hidden_size)
        elif hasattr(cli_args, "hidden_layers"):
            self.hidden_layers = list(cli_args.hidden_layers)
        else:
            self.hidden_layers = [128, 128]

        self.activation_name = getattr(cli_args, "activation", "relu")
        self.loss_name = get_loss(getattr(cli_args, "loss", "cross_entropy"))
        self.weight_init = getattr(cli_args, "weight_init", "xavier")

        self.learning_rate = getattr(cli_args, "learning_rate", 0.001)
        self.weight_decay = getattr(cli_args, "weight_decay", 0.0)
        self.optimizer_name = getattr(cli_args, "optimizer", "sgd")

        self.layers = []
        prev_size = self.input_size

        for hidden_size in self.hidden_layers:
            layer = NeuralLayer(
                input_size=prev_size,
                output_size=hidden_size,
                activation=self.activation_name,
                weight_init=self.weight_init,
            )
            self.layers.append(layer)
            prev_size = hidden_size

        if self.weight_init == "xavier":
            limit = np.sqrt(6.0 / (prev_size + self.output_size))
            self.output_W = np.random.uniform(
                -limit, limit, (prev_size, self.output_size)
            ).astype(np.float64)
        elif self.weight_init == "random":
            self.output_W = (0.01 * np.random.randn(prev_size, self.output_size)).astype(
                np.float64
            )
        elif self.weight_init == "zeros":
            self.output_W = np.zeros((prev_size, self.output_size), dtype=np.float64)
        else:
            raise ValueError("weight_init must be one of: random, xavier, zeros")

        self.output_b = np.zeros((1, self.output_size), dtype=np.float64)

        self.output_input = None
        self.logits = None
        self.probs = None

        self.output_grad_W = None
        self.output_grad_b = None

        self.grad_W = None
        self.grad_b = None

        self.optimizer = get_optimizer(
            name=self.optimizer_name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied).

        X shape: (batch_size, input_dim)
        output shape: (batch_size, output_dim)
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)

        self.output_input = out
        self.logits = np.dot(out, self.output_W) + self.output_b
        return self.logits

    def backward(self, y_true, y_pred=None):
        """
        Backward propagation to compute gradients.

        Returns gradients in forward layer order:
        grad_W[0] corresponds to W0,
        grad_W[1] corresponds to W1,
        ...
        grad_W[-1] corresponds to the output layer weights.
        """
        if self.loss_name == "cross_entropy":
            _, probs = cross_entropy_loss(self.logits, y_true)
            self.probs = probs
            grad_logits = cross_entropy_grad(probs, y_true)

        elif self.loss_name == "mse":
            probs = softmax(self.logits)
            self.probs = probs
            grad_probs = mse_grad(probs, y_true)

            batch_size = probs.shape[0]
            grad_logits = np.zeros_like(probs)

            for i in range(batch_size):
                p = probs[i].reshape(-1, 1)
                jacobian = np.diagflat(p) - np.dot(p, p.T)
                grad_logits[i] = np.dot(jacobian, grad_probs[i])
        else:
            raise ValueError("Unsupported loss")

        # Output layer gradients
        self.output_grad_W = np.dot(self.output_input.T, grad_logits)
        self.output_grad_b = np.sum(grad_logits, axis=0, keepdims=True)

        # Backpropagate into hidden layers
        grad_hidden = np.dot(grad_logits, self.output_W.T)

        for layer in reversed(self.layers):
            grad_hidden = layer.backward(grad_hidden)

        # Store gradients in forward order to match get_weights():
        # W0, W1, ..., W_last_hidden, W_output
        grad_W_list = [layer.grad_W for layer in self.layers] + [self.output_grad_W]
        grad_b_list = [layer.grad_b for layer in self.layers] + [self.output_grad_b]

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step(self)

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        n = X_train.shape[0]

        for _ in range(epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)

            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                self.forward(X_batch)
                self.backward(y_batch)
                self.update_weights()

    def evaluate(self, X, y):
        logits = self.forward(X)
        probs = softmax(logits)

        if self.loss_name == "cross_entropy":
            loss, _ = cross_entropy_loss(logits, y)
        else:
            loss = mse_loss(probs, y)

        preds = np.argmax(probs, axis=1)
        true = np.argmax(y, axis=1)
        accuracy = float(np.mean(preds == true))

        return {
            "logits": logits,
            "loss": float(loss),
            "accuracy": accuracy,
            "predictions": preds,
            "probabilities": probs,
        }

    def get_weights(self):
        d = {}

        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()

        d[f"W{len(self.layers)}"] = self.output_W.copy()
        d[f"b{len(self.layers)}"] = self.output_b.copy()

        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"

            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

        out_w_key = f"W{len(self.layers)}"
        out_b_key = f"b{len(self.layers)}"

        if out_w_key in weight_dict:
            self.output_W = weight_dict[out_w_key].copy()
        if out_b_key in weight_dict:
            self.output_b = weight_dict[out_b_key].copy()

    def save_model(self, model_path, config_path=None):
        payload = {
            "config": {
                "dataset": self.dataset,
                "input_size": self.input_size,
                "output_size": self.output_size,
                "hidden_layers": self.hidden_layers,
                "activation": self.activation_name,
                "loss": self.loss_name,
                "weight_init": self.weight_init,
                "optimizer": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
            "weights": self.get_weights(),
        }

        np.save(model_path, payload, allow_pickle=True)

        if config_path is not None:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(payload["config"], f, indent=4)

    @classmethod
    def load_model(cls, model_path):
        payload = np.load(model_path, allow_pickle=True).item()
        config = payload["config"]

        class Args:
            pass

        args = Args()
        args.dataset = config["dataset"]
        args.hidden_size = config["hidden_layers"]
        args.activation = config["activation"]
        args.loss = config["loss"]
        args.weight_init = config["weight_init"]
        args.optimizer = config["optimizer"]
        args.learning_rate = config["learning_rate"]
        args.weight_decay = config["weight_decay"]

        model = cls(args)
        model.set_weights(payload["weights"])
        return model