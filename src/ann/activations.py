import numpy as np

class Sigmoid:
    @staticmethod
    def forward(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def backward(output):
        return output * (1.0 - output)

class Tanh:
    
    @staticmethod
    def forward(x):
        return np.tanh(x)
    
    @staticmethod
    def backward(output):
        return 1.0 - output ** 2

class ReLU:
    
    @staticmethod
    def forward(x):
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x):
        return (x > 0).astype(np.float32)

class Softmax:

    @staticmethod
    def forward(x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def get_activation(name):
    activations = {
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'relu': ReLU
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name.lower()]