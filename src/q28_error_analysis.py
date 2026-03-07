"""
Question 2.8 - Error Analysis
Confusion Matrix + Misclassified 
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot


def parse_arguments():
    parser = argparse.ArgumentParser(description="Question 2.8 Error Analysis")
    parser.add_argument("--model_path", type=str, default="src/best_model.npy")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--num_images", type=int, default=20)
    return parser.parse_args()


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.show()


def show_misclassified_images(X, y_true, y_pred, num_images=20):
    mis_idx = np.where(y_true != y_pred)[0]

    if len(mis_idx) == 0:
        print("No misclassified images found.")
        return

    num_images = min(num_images, len(mis_idx))
    cols = 5
    rows = int(np.ceil(num_images / cols))

    plt.figure(figsize=(12, 2.5 * rows))

    for i in range(num_images):
        idx = mis_idx[i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap="gray")
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}")
        plt.axis("off")

    plt.suptitle("Misclassified Test Images", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_arguments()

    _, _, X_test, y_test = load_data(args.dataset)
    y_test_oh = one_hot(y_test, 10)

    model = NeuralNetwork.load_model(args.model_path)
    result = model.evaluate(X_test, y_test_oh)
    y_pred = result["predictions"]

    cm = compute_confusion_matrix(y_test, y_pred, num_classes=10)

    print("Test Accuracy:", result["accuracy"])
    print("Test Loss:", result["loss"])

    plot_confusion_matrix(cm, title=f"Confusion Matrix - {args.dataset}")
    show_misclassified_images(X_test, y_test, y_pred, num_images=args.num_images)


if __name__ == "__main__":
    main()