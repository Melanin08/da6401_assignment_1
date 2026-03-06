"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from ann.neural_layer import DenseLayer
from ann.activations import get_activation, Softmax


class NeuralNetwork:
    """
    Multi-Layer Perceptron for classification
    """
    def __init__(self, input_size, hidden_layers, output_size, 
                 activation='relu', weight_init='xavier', use_softmax=True):
        """
        Initialize neural network
        Args:
            input_size: dimension of input features
            hidden_layers: list of hidden layer sizes (e.g., [128, 64, 32])
            output_size: number of output classes
            activation: activation function for hidden layers
            weight_init: weight initialization method ('xavier' or 'random')
            use_softmax: whether to apply softmax to output
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_name = activation
        self.weight_init = weight_init
        self.use_softmax = use_softmax
        
        # Get activation function
        self.activation_fn = get_activation(activation)
        
        # Build layers
        self.layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(layer_sizes[i], layer_sizes[i+1], weight_init)
            self.layers.append(layer)
        
        # Store activations for backward pass
        self.activations = []
        self.z_values = []  # pre-activation values
        
        # Store gradients
        self.grad_W = None
        self.grad_b = None
    
    def forward(self, X):
        """
        Forward propagation through all layers
        Args:
            X: input (batch_size, input_size)
        Returns:
            output: (batch_size, output_size) - logits (no softmax)
        """
        self.activations = [X]
        self.z_values = []
        
        # Forward through hidden layers with activation
        a = X
        for i, layer in enumerate(self.layers[:-1]):
            z = layer.forward(a)
            self.z_values.append(z)
            a = self.activation_fn.forward(z)
            self.activations.append(a)
        
        # Output layer (no activation)
        z = self.layers[-1].forward(a)
        self.z_values.append(z)
        
        return z
    
    def backward(self, dL_dZ):
        """
        Backward propagation through all layers
        Args:
            dL_dZ: gradient from loss function w.r.t. final layer output
        Returns:
            grad_W, grad_b: lists of gradients for each layer
        """
        grad_W_list = []
        grad_b_list = []
        
        # Backprop through output layer
        delta = dL_dZ
        for i in range(len(self.layers) - 1, -1, -1):
            # Backward pass through layer
            dL_dX = self.layers[i].backward(delta)
            grad_W_list.insert(0, self.layers[i].grad_W)
            grad_b_list.insert(0, self.layers[i].grad_b)
            
            # Backprop through activation (except for output layer)
            if i > 0:
                delta = dL_dX * self.activation_fn.backward(self.activations[i])
            else:
                delta = dL_dX
        
        # Store gradients as object arrays
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb
        
        return self.grad_W, self.grad_b
    
    def update_weights(self, optimizer):
        """
        Update weights using optimizer
        Args:
            optimizer: optimizer instance with update method
        """
        for i, layer in enumerate(self.layers):
            dW = optimizer.update(layer.W, self.grad_W[i], param_id=f'W{i}')
            db = optimizer.update(layer.b, self.grad_b[i], param_id=f'b{i}')
            layer.update_weights(dW, db)
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs, batch_size, optimizer, loss_fn, 
              verbose=True, wandb_log=None):
        """
        Training loop
        Args:
            X_train: training data
            y_train: training labels (one-hot encoded)
            X_val: validation data
            y_val: validation labels (one-hot encoded)
            epochs: number of epochs
            batch_size: batch size
            optimizer: optimizer instance
            loss_fn: loss function instance
            verbose: print progress
            wandb_log: wandb logging function
        """
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            train_loss = 0
            train_acc = 0
            num_batches = 0
            
            # Mini-batch training
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Forward pass
                logits = self.forward(X_batch)
                
                # Compute loss
                loss = loss_fn.forward(logits, y_batch)
                train_loss += loss
                
                # Backward pass
                grad = loss_fn.backward()
                self.backward(grad)
                
                # Update weights
                self.update_weights(optimizer)
                
                # Compute accuracy
                predictions = np.argmax(logits, axis=1)
                labels = np.argmax(y_batch, axis=1)
                batch_acc = np.mean(predictions == labels)
                train_acc += batch_acc
                
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            avg_train_acc = train_acc / num_batches
            
            # Validation
            val_logits = self.forward(X_val)
            val_loss = loss_fn.forward(val_logits, y_val)
            val_predictions = np.argmax(val_logits, axis=1)
            val_labels = np.argmax(y_val, axis=1)
            val_acc = np.mean(val_predictions == val_labels)
            
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Log to wandb
            if wandb_log:
                wandb_log({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'train_accuracy': avg_train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                })
    
    def predict(self, X):
        """
        Make predictions
        Args:
            X: input data (batch_size, input_size)
        Returns:
            predictions: class predictions (batch_size,)
            probabilities: class probabilities (batch_size, output_size)
        """
        logits = self.forward(X)
        
        if self.use_softmax:
            probabilities = Softmax.forward(logits)
        else:
            # For regression-style output
            probabilities = logits
        
        predictions = np.argmax(logits, axis=1)
        return predictions, probabilities
    
    def evaluate(self, X, y):
        """
        Evaluate on dataset
        Args:
            X: input data
            y: labels (one-hot encoded)
        Returns:
            accuracy: classification accuracy
        """
        predictions, _ = self.predict(X)
        labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy
    
    def get_weights(self):
        """Get all weights and biases"""
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            weights_dict[f'W{i}'] = layer.W.copy()
            weights_dict[f'b{i}'] = layer.b.copy()
        return weights_dict
    
    def set_weights(self, weights_dict):
        """Set all weights and biases"""
        for i, layer in enumerate(self.layers):
            if f'W{i}' in weights_dict:
                layer.W = weights_dict[f'W{i}'].copy()
            if f'b{i}' in weights_dict:
                layer.b = weights_dict[f'b{i}'].copy()


