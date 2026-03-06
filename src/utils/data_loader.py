"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist


def load_data(dataset_name):
    """
    Load dataset and preprocess images.

    Args:
        dataset_name (str): 'mnist' or 'fashion_mnist'

    Returns:
        x_train, y_train, x_test, y_test
    """

    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    # Flatten images (28x28 → 784)
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    # Normalize pixel values
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test


def one_hot(y, num_classes=10):
    """
    Convert labels to one-hot encoding
    """
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded