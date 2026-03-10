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
        self.dataset = getattr(cli_args, "dataset", None) or "mnist"
        self.input_size = 784
        self.output_size = 10

        hidden_size = getattr(cli_args, "hidden_size", None)
        hidden_layers = getattr(cli_args, "hidden_layers", None)

        if hidden_size is not None:
            self.hidden_layers = list(hidden_size)
        elif hidden_layers is not None:
            self.hidden_layers = list(hidden_layers)
        else:
            self.hidden_layers = [128, 128]

        self.activation_name = getattr(cli_args, "activation", None) or "relu"
        self.loss_name = get_loss(getattr(cli_args, "loss", None) or "cross_entropy")
        self.weight_init = getattr(cli_args, "weight_init", None) or "xavier"

        self.learning_rate = getattr(cli_args, "learning_rate", 0.001)
        self.weight_decay = getattr(cli_args, "weight_decay", 0.0)
        self.optimizer_name = getattr(cli_args, "optimizer", None) or "sgd"

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
            self.output_W = (0.01 * np.random.randn(prev_size, self.output_size)).astype(np.float64)
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
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
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
        Returns two numpy arrays: grad_Ws, grad_bs.
        - grad_Ws[0] is gradient for the last (output) layer weights,
          grad_bs[0] is gradient for the last layer biases, and so on.
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

        self.output_grad_W = np.dot(self.output_input.T, grad_logits)
        self.output_grad_b = np.sum(grad_logits, axis=0, keepdims=True)

        grad_hidden = np.dot(grad_logits, self.output_W.T)

        grad_W_list = [self.output_grad_W]
        grad_b_list = [self.output_grad_b]

        for layer in reversed(self.layers):
            grad_hidden = layer.backward(grad_hidden)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

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

                logits = self.forward(X_batch)
                _ = logits
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
        args.dataset = config.get("dataset", "mnist")
        args.hidden_size = config.get("hidden_layers", [128, 128])
        args.activation = config.get("activation", "relu")
        args.loss = config.get("loss", "cross_entropy")
        args.weight_init = config.get("weight_init", "xavier")
        args.optimizer = config.get("optimizer", "sgd")
        args.learning_rate = config.get("learning_rate", 0.001)
        args.weight_decay = config.get("weight_decay", 0.0)

        model = cls(args)
        model.set_weights(payload["weights"])
        return model
