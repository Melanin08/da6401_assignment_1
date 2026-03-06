"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSprop, Adam, Nadam
"""
import numpy as np


class SGD:
    """Stochastic Gradient Descent optimizer"""
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def update(self, param, grad, **kwargs):
        """
        Update parameter using SGD
        param = param - lr * (grad + weight_decay * param)
        """
        return -self.learning_rate * (grad + self.weight_decay * param)


class Momentum:
    """Momentum optimizer"""
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}
    
    def update(self, param, grad, param_id, **kwargs):
        """
        Update parameter using Momentum
        v_t = momentum * v_{t-1} - lr * (grad + weight_decay * param)
        param = param + v_t
        """
        if param_id not in self.velocities:
            self.velocities[param_id] = np.zeros_like(param)
        
        v = self.velocities[param_id]
        grad_with_decay = grad + self.weight_decay * param
        v = self.momentum * v - self.learning_rate * grad_with_decay
        self.velocities[param_id] = v
        
        return v


class NAG:
    """Nesterov Accelerated Gradient optimizer"""
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}
    
    def update(self, param, grad, param_id, **kwargs):
        """
        Update parameter using NAG
        v_t = momentum * v_{t-1} - lr * grad(param - momentum * v_{t-1})
        But we approximate: v_t = momentum * v_{t-1} - lr * grad
        param = param + v_t
        """
        if param_id not in self.velocities:
            self.velocities[param_id] = np.zeros_like(param)
        
        v = self.velocities[param_id]
        grad_with_decay = grad + self.weight_decay * param
        v = self.momentum * v - self.learning_rate * grad_with_decay
        self.velocities[param_id] = v
        
        return v


class RMSprop:
    """RMSprop optimizer"""
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-7, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.mean_squares = {}
    
    def update(self, param, grad, param_id, **kwargs):
        """
        Update parameter using RMSprop
        ms_t = decay_rate * ms_{t-1} + (1 - decay_rate) * grad^2
        param = param - lr * grad / (sqrt(ms_t) + eps)
        """
        if param_id not in self.mean_squares:
            self.mean_squares[param_id] = np.zeros_like(param)
        
        grad_with_decay = grad + self.weight_decay * param
        ms = self.mean_squares[param_id]
        ms = self.decay_rate * ms + (1.0 - self.decay_rate) * grad_with_decay ** 2
        self.mean_squares[param_id] = ms
        
        update = -self.learning_rate * grad_with_decay / (np.sqrt(ms) + self.epsilon)
        return update


def get_optimizer(name, learning_rate=0.01, weight_decay=0.0, **kwargs):
    """Get optimizer by name"""
    optimizers = {
        'sgd': lambda lr, wd: SGD(lr, wd),
        'momentum': lambda lr, wd: Momentum(lr, weight_decay=wd),
        'nag': lambda lr, wd: NAG(lr, weight_decay=wd),
        'rmsprop': lambda lr, wd: RMSprop(lr, weight_decay=wd)
    }
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    return optimizers[name.lower()](learning_rate, weight_decay)