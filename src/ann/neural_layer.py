import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, weight_init='xavier'):
       
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights and biases
        if weight_init.lower() == 'xavier':
            # Xavier initialization: sample from U(-limit, limit)
            # where limit = sqrt(6 / (input_size + output_size))
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        else:  # random
            # Random initialization from standard normal
            self.W = np.random.randn(input_size, output_size) * 0.01
        
        # Initialize biases to zero
        self.b = np.zeros((1, output_size))
        
        # Store gradients
        self.grad_W = None
        self.grad_b = None
        
        # Store inputs for backward pass
        self.X = None
    
    def forward(self, X):
        """
        Forward pass
        Args:
            X: input (batch_size, input_size)
        Returns:
            output: (batch_size, output_size)
        """
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, dL_dZ):
        """
        Backward pass: compute gradients
        Args:
            dL_dZ: gradient of loss w.r.t. layer output (batch_size, output_size)
        Returns:
            dL_dX: gradient w.r.t. input (batch_size, input_size)
        """
        batch_size = self.X.shape[0]
        
        # Gradient w.r.t. weights: dL/dW = X^T @ dL/dZ
        self.grad_W = (self.X.T @ dL_dZ) / batch_size
        
        # Gradient w.r.t. biases: dL/db = sum(dL/dZ) over batch
        self.grad_b = np.sum(dL_dZ, axis=0, keepdims=True) / batch_size
        
        # Gradient w.r.t. input: dL/dX = dL/dZ @ W^T
        dL_dX = dL_dZ @ self.W.T
        
        return dL_dX
    
    def update_weights(self, dW, db):
        self.W += dW
        self.b += db