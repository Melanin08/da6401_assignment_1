
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

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_name = activation
        self.weight_init = weight_init
        self.use_softmax = use_softmax

        # Activation function
        self.activation_fn = get_activation(activation)

        # Build layers
        self.layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(layer_sizes[i], layer_sizes[i+1], weight_init)
            self.layers.append(layer)

        self.activations = []
        self.z_values = []

        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        Forward propagation
        """
        self.activations = [X]
        self.z_values = []

        a = X

        # Hidden layers
        for layer in self.layers[:-1]:
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
        Backpropagation
        """

        grad_W_list = []
        grad_b_list = []

        delta = dL_dZ

        for i in range(len(self.layers) - 1, -1, -1):

            # Backward through Dense layer
            dL_dX = self.layers[i].backward(delta)

            grad_W_list.insert(0, self.layers[i].grad_W)
            grad_b_list.insert(0, self.layers[i].grad_b)

            # Apply activation derivative for hidden layers
            if i > 0:
                if self.activation_name.lower() == 'relu':
                    delta = dL_dX * self.activation_fn.backward(self.z_values[i-1])
                else:
                    delta = dL_dX * self.activation_fn.backward(self.activations[i])
            else:
                delta = dL_dX

        # Convert to object arrays
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i in range(len(grad_W_list)):
            self.grad_W[i] = grad_W_list[i]
            self.grad_b[i] = grad_b_list[i]

        return self.grad_W, self.grad_b

    def update_weights(self, optimizer):

        for i, layer in enumerate(self.layers):
            dW = optimizer.update(layer.W, self.grad_W[i], param_id=f'W{i}')
            db = optimizer.update(layer.b, self.grad_b[i], param_id=f'b{i}')

            layer.update_weights(dW, db)

    def train(self, X_train, y_train, X_val, y_val,
              epochs, batch_size, optimizer, loss_fn,
              verbose=True, wandb_log=None):

        num_samples = X_train.shape[0]

        for epoch in range(epochs):

            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            train_loss = 0
            train_acc = 0
            grad_norms = []
            activation_means = {}
            activation_stds = {}
            dead_neuron_counts = {}
            num_batches = 0

            for batch_start in range(0, num_samples, batch_size):

                batch_end = min(batch_start + batch_size, num_samples)

                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]

                # Forward
                logits = self.forward(X_batch)

                # Loss
                loss = loss_fn.forward(logits, y_batch)
                train_loss += loss

                # Backward
                grad = loss_fn.backward()
                self.backward(grad)

                # Compute activation statistics for monitoring dead neurons
                activation_stats = {}
                dead_neurons = {}
                for layer_idx in range(len(self.layers) - 1):  # hidden layers only
                    activations = self.activations[layer_idx + 1]  # activations[1] is first hidden layer
                    activation_stats[f'layer_{layer_idx}_mean'] = np.mean(activations)
                    activation_stats[f'layer_{layer_idx}_std'] = np.std(activations)
                    # Count dead neurons (neurons that output 0 for all samples in batch)
                    dead_count = np.sum(np.all(activations == 0, axis=0))
                    dead_neurons[f'layer_{layer_idx}_dead_neurons'] = dead_count
                
                # Accumulate stats
                for key, value in activation_stats.items():
                    if key not in activation_means:
                        activation_means[key] = []
                    activation_means[key].append(value)
                
                for key, value in dead_neurons.items():
                    if key not in dead_neuron_counts:
                        dead_neuron_counts[key] = []
                    dead_neuron_counts[key].append(value)

                # Update
                self.update_weights(optimizer)

                # Accuracy
                predictions = np.argmax(logits, axis=1)
                labels = np.argmax(y_batch, axis=1)

                batch_acc = np.mean(predictions == labels)

                train_acc += batch_acc
                num_batches += 1

            avg_train_loss = train_loss / num_batches
            avg_train_acc = train_acc / num_batches
            avg_grad_norm = np.mean(grad_norms)
            
            # Average activation stats across batches
            avg_activation_stats = {}
            for key, values in activation_means.items():
                avg_activation_stats[key] = np.mean(values)
            for key, values in activation_stds.items():
                avg_activation_stats[key] = np.mean(values)
            for key, values in dead_neuron_counts.items():
                avg_activation_stats[key] = np.mean(values)

            # Validation
            val_logits = self.forward(X_val)

            val_loss = loss_fn.forward(val_logits, y_val)

            val_predictions = np.argmax(val_logits, axis=1)
            val_labels = np.argmax(y_val, axis=1)

            val_acc = np.mean(val_predictions == val_labels)

            if verbose:
                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

            if wandb_log:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'train_accuracy': avg_train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'first_layer_grad_norm': avg_grad_norm
                }
                log_dict.update(avg_activation_stats)
                wandb_log(log_dict)

    def predict(self, X):

        logits = self.forward(X)

        if self.use_softmax:
            softmax = Softmax()
            probabilities = softmax.forward(logits)
        else:
            probabilities = logits

        predictions = np.argmax(logits, axis=1)

        return predictions, probabilities

    def evaluate(self, X, y):

        predictions, _ = self.predict(X)

        labels = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == labels)

        return accuracy

    def get_weights(self):

        weights_dict = {}

        for i, layer in enumerate(self.layers):
            weights_dict[f'W{i}'] = layer.W.copy()
            weights_dict[f'b{i}'] = layer.b.copy()

        return weights_dict

    def set_weights(self, weights_dict):

        for i, layer in enumerate(self.layers):

            if f'W{i}' in weights_dict:
                layer.W = weights_dict[f'W{i}'].copy()

            if f'b{i}' in weights_dict:
                layer.b = weights_dict[f'b{i}'].copy()

