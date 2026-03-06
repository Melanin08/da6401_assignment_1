"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
from ann.activations import Softmax


class CrossEntropyLoss:
    """
    Cross-Entropy Loss for multi-class classification
    Used with softmax output layer
    Loss = -sum(y_true * log(y_pred))
    """
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def forward(self, logits, y_true):
        """
        Forward pass: compute cross-entropy loss
        Args:
            logits: raw output from network (batch_size, num_classes)
            y_true: one-hot encoded labels (batch_size, num_classes)
        Returns:
            scalar loss value
        """
        # Apply softmax to get probabilities
        self.y_pred = Softmax.forward(logits)
        self.y_true = y_true
        
        # Clip predictions to avoid log(0)
        epsilon = 1e-7
        self.y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)
        
        # Compute cross-entropy: -sum(y_true * log(y_pred))
        batch_size = y_true.shape[0]
        loss = -np.sum(y_true * np.log(self.y_pred)) / batch_size
        
        return loss
    
    def backward(self):
        """
        Backward pass: compute gradient
        For cross-entropy with softmax: dL/dz = (y_pred - y_true) / batch_size
        """
        batch_size = self.y_true.shape[0]
        grad = (self.y_pred - self.y_true) / batch_size
        return grad


class MSELoss:
    """
    Mean Squared Error Loss
    Loss = (1/2n) * sum((y_pred - y_true)^2)
    """
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        """
        Forward pass: compute MSE loss
        Args:
            y_pred: network predictions (batch_size, num_classes)
            y_true: one-hot encoded labels (batch_size, num_classes)
        Returns:
            scalar loss value
        """
        self.y_pred = y_pred
        self.y_true = y_true
        
        # MSE = mean((y_pred - y_true)^2)
        batch_size = y_true.shape[0]
        loss = np.sum((y_pred - y_true) ** 2) / (2 * batch_size)
        
        return loss
    
    def backward(self):
        """
        Backward pass: compute gradient
        For MSE: dL/dz = (y_pred - y_true) / batch_size
        """
        batch_size = self.y_true.shape[0]
        grad = (self.y_pred - self.y_true) / batch_size
        return grad


def get_loss_function(name):
    """Get loss function by name"""
    loss_functions = {
        'cross_entropy': CrossEntropyLoss,
        'mse': MSELoss
    }
    if name.lower() not in loss_functions:
        raise ValueError(f"Unknown loss function: {name}")
    return loss_functions[name.lower()]