import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist


MNIST_CLASS_NAMES = [str(i) for i in range(10)]

FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def load_data(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("dataset_name must be 'mnist' or 'fashion_mnist'")

    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float64) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return x_train, y_train, x_test, y_test


def one_hot_encode(y, num_classes=10):
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def one_hot(y, num_classes=10):
    return one_hot_encode(y, num_classes)


def decode_one_hot(y_one_hot):
    return np.argmax(y_one_hot, axis=1)


def get_class_names(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "mnist":
        return MNIST_CLASS_NAMES
    elif dataset_name == "fashion_mnist":
        return FASHION_MNIST_CLASS_NAMES
    else:
        raise ValueError("dataset_name must be 'mnist' or 'fashion_mnist'")