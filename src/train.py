"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import os
import json
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-o", "--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[128, 128])
    parser.add_argument("-a", "--activation", type=str, default="relu",
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-wi", "-w_i", "--weight_init", type=str, default="xavier",
                        choices=["random", "xavier", "zeros"])
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment1")
    parser.add_argument("--model_save_path", type=str, default="src/best_model.npy")
    parser.add_argument("--config_save_path", type=str, default="src/best_config.json")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def precision_recall_f1_macro(y_true, y_pred, num_classes=10):
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def train_val_split(X, y, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split = int((1.0 - val_ratio) * X.shape[0])
    train_idx = indices[:split]
    val_idx = indices[split:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def get_batches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def train(args, init_wandb=True):
    if len(args.hidden_size) != args.num_layers:
        raise ValueError("num_layers must match the number of values passed to hidden_size")

    np.random.seed(args.seed)

    if init_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            settings=wandb.Settings(init_timeout=240)
        )

    X_train_full, y_train_full, X_test, y_test = load_data(args.dataset)

    X_train, y_train, X_val, y_val = train_val_split(
        X_train_full, y_train_full, val_ratio=0.1, seed=args.seed
    )

    y_train_oh = one_hot(y_train, 10)
    y_val_oh = one_hot(y_val, 10)
    y_test_oh = one_hot(y_test, 10)

    model = NeuralNetwork(args)

    best_test_f1 = -1.0

    for epoch in range(args.epochs):
        batch_losses = []

        for X_batch, y_batch in get_batches(X_train, y_train_oh, args.batch_size):
            logits = model.forward(X_batch)

            # Log hidden layer activations for dead neuron investigation
            for i, layer in enumerate(model.layers):
                activation_values = layer.a
                wandb.log({
                    f"layer_{i+1}_activation_mean": float(np.mean(activation_values)),
                    f"layer_{i+1}_activation_std": float(np.std(activation_values)),
                    f"layer_{i+1}_zero_fraction": float(np.mean(activation_values == 0.0)),
                })

            if model.loss_name == "cross_entropy":
                from ann.objective_functions import cross_entropy_loss
                loss, _ = cross_entropy_loss(logits, y_batch)
            else:
                from ann.activations import softmax
                from ann.objective_functions import mse_loss
                loss = mse_loss(softmax(logits), y_batch)

            batch_losses.append(loss)

            model.backward(y_batch, None)

            # Q2.9: log gradients of 5 neurons in the first hidden layer
            # We use the mean gradient value for each neuron's incoming weights
            grad_matrix = model.layers[0].grad_W  # shape: (input_dim, hidden_dim)

            num_neurons_to_log = min(5, grad_matrix.shape[1])
            for i in range(num_neurons_to_log):
                neuron_grad_value = float(np.mean(grad_matrix[:, i]))
                wandb.log({f"neuron_{i+1}_grad": neuron_grad_value})

            model.update_weights()

        train_eval = model.evaluate(X_train, y_train_oh)
        val_eval = model.evaluate(X_val, y_val_oh)
        test_eval = model.evaluate(X_test, y_test_oh)

        train_pred = train_eval["predictions"]
        val_pred = val_eval["predictions"]
        test_pred = test_eval["predictions"]

        train_precision, train_recall, train_f1 = precision_recall_f1_macro(y_train, train_pred)
        val_precision, val_recall, val_f1 = precision_recall_f1_macro(y_val, val_pred)
        test_precision, test_recall, test_f1 = precision_recall_f1_macro(y_test, test_pred)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_eval["loss"],
            "val_loss": val_eval["loss"],
            "test_loss": test_eval["loss"],
            "train_accuracy": train_eval["accuracy"],
            "val_accuracy": val_eval["accuracy"],
            "test_accuracy": test_eval["accuracy"],
            "train_f1": train_f1,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "train_precision": train_precision,
            "val_precision": val_precision,
            "test_precision": test_precision,
            "train_recall": train_recall,
            "val_recall": val_recall,
            "test_recall": test_recall,
            "batch_loss_mean": float(np.mean(batch_losses)),
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Acc: {train_eval['accuracy']:.6f} | "
            f"Val Acc: {val_eval['accuracy']:.6f} | "
            f"Test Acc: {test_eval['accuracy']:.6f} | "
            f"Test F1: {test_f1:.6f}"
        )

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1

            os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(args.config_save_path), exist_ok=True)

            model.save_model(args.model_save_path, args.config_save_path)

            best_config = {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "activation": args.activation,
                "loss": args.loss,
                "weight_init": args.weight_init,
                "best_test_f1": best_test_f1,
                "best_test_accuracy": test_eval["accuracy"],
                "best_test_precision": test_precision,
                "best_test_recall": test_recall,
            }

            with open(args.config_save_path, "w", encoding="utf-8") as f:
                json.dump(best_config, f, indent=4)

    if init_wandb:
        wandb.finish()

    print("Training complete!")


def main():
    args = parse_arguments()
    train(args, init_wandb=True)


if __name__ == "__main__":
    main()