"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
import os
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--model_path", type=str, default="src/best_model.npy")
    parser.add_argument("-d", "--dataset", type=str, default=None, choices=["mnist", "fashion_mnist"])
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-a", "--activation", type=str, default=None)
    parser.add_argument("-nhl", "--hidden_layers", nargs="*", type=int, default=None)

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    return NeuralNetwork.load_model(model_path)


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


def infer_dataset_from_model(model_path):
    """
    Try to infer dataset from saved model config.
    """
    if not os.path.exists(model_path):
        return "mnist"

    payload = np.load(model_path, allow_pickle=True).item()
    config = payload.get("config", {})
    return config.get("dataset", "mnist")


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.

    Returns Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    y_test_oh = one_hot(y_test, 10)
    result = model.evaluate(X_test, y_test_oh)
    y_pred = result["predictions"]

    precision, recall, f1 = precision_recall_f1_macro(y_test, y_pred, num_classes=10)

    return {
        "logits": result["logits"],
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    """
    Main inference function.
    """
    args = parse_arguments()

    if args.dataset is None:
        args.dataset = infer_dataset_from_model(args.model_path)

    _, _, X_test, y_test = load_data(args.dataset)
    model = load_model(args.model_path)
    metrics = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1-score: {metrics['f1']:.6f}")
    print(f"Loss: {metrics['loss']:.6f}")

    return metrics


if __name__ == "__main__":
    main()