"""
Inference Script
Evaluate trained models on test sets
"""
import argparse
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from utils.data_loader import load_data, one_hot
from ann.neural_network import NeuralNetwork

# Manual metric computations using numpy

def accuracy_score_np(labels, preds):
    return np.mean(labels == preds)

def precision_score_np(labels, preds, average='weighted'):
    # calculate per-class precision and average
    classes = np.unique(labels)
    precisions = []
    for c in classes:
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        precisions.append(tp / (tp + fp + 1e-8))
    if average == 'weighted':
        weights = [(labels == c).sum() for c in classes]
        return np.sum(np.array(precisions) * np.array(weights)) / np.sum(weights)
    return np.mean(precisions)

def recall_score_np(labels, preds, average='weighted'):
    classes = np.unique(labels)
    recalls = []
    for c in classes:
        tp = np.sum((preds == c) & (labels == c))
        fn = np.sum((preds != c) & (labels == c))
        recalls.append(tp / (tp + fn + 1e-8))
    if average == 'weighted':
        weights = [(labels == c).sum() for c in classes]
        return np.sum(np.array(recalls) * np.array(weights)) / np.sum(weights)
    return np.mean(recalls)

def f1_score_np(labels, preds, average='weighted'):
    p = precision_score_np(labels, preds, average=average)
    r = recall_score_np(labels, preds, average=average)
    return 2 * p * r / (p + r + 1e-8)

def confusion_matrix_np(labels, preds):
    classes = np.unique(labels)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for i, c in enumerate(classes):
        for j, d in enumerate(classes):
            cm[i, j] = np.sum((labels == c) & (preds == d))
    return cm


def parse_arguments():
    """Parse command-line arguments for inference"""
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument(
        "-model_path", "--model_path",
        type=str,
        required=True,
        help="Path to saved model weights (.npy file)"
    )
    
    parser.add_argument(
        "-config_path", "--config_path",
        type=str,
        help="Path to model config file (.json)"
    )
    
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        choices=["mnist", "fashion_mnist"],
        help="Dataset to evaluate on"
    )
    
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "-a", "--activation",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh"],
        help="Activation function"
    )
    
    parser.add_argument(
        "-sz", "--hidden_size",
        type=int,
        nargs="+",
        required=True,
        help="Hidden layer sizes"
    )
    
    parser.add_argument(
        "-w_i", "--weight_init",
        type=str,
        default="xavier",
        choices=["random", "xavier"],
        help="Weight initialization method"
    )
    
    return parser.parse_args()


def load_model_and_config(model_path, config_path=None):
    """Load trained model weights and config"""
    # Load weights
    weights = np.load(model_path, allow_pickle=True).item()
    
    # Load config if provided
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return weights, config


def create_model(weights, hidden_layers, activation, weight_init):
    """Create model and load weights"""
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=hidden_layers,
        output_size=10,
        activation=activation,
        weight_init=weight_init,
        use_softmax=True
    )
    
    # Load weights
    model.set_weights(weights)
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data
    
    Returns:
        Dictionary with logits, accuracy, precision, recall, f1
    """
    # Get predictions
    logits = model.forward(X_test)
    predictions = np.argmax(logits, axis=1)
    labels = np.argmax(y_test, axis=1)
    
    # Calculate metrics manually
    accuracy = accuracy_score_np(labels, predictions)
    precision = precision_score_np(labels, predictions, average='weighted')
    recall = recall_score_np(labels, predictions, average='weighted')
    f1 = f1_score_np(labels, predictions, average='weighted')
    
    # Get confusion matrix
    cm = confusion_matrix_np(labels, predictions)
    
    results = {
        'logits': logits,
        'predictions': predictions,
        'labels': labels,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return results


def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print("="*50 + "\n")


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    classes = np.arange(10)
    tick_marks = np.arange(10)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def main():
    """Main inference function"""
    args = parse_arguments()
    
    print(f"Loading model from {args.model_path}...")
    
    # Load model and config
    weights, config = load_model_and_config(args.model_path, args.config_path)
    
    # Use config values if available, otherwise use arguments
    hidden_size = config.get('hidden_size', args.hidden_size)
    activation = config.get('activation', args.activation)
    weight_init = config.get('weight_init', args.weight_init)
    dataset = config.get('dataset', args.dataset)
    
    # Create model with saved weights
    print("Creating model...")
    model = create_model(weights, hidden_size, activation, weight_init)
    
    # Load test data
    print(f"Loading {dataset} dataset...")
    X_train, y_train, X_test, y_test = load_data(dataset)
    
    # One-hot encode labels
    y_test = one_hot(y_test)
    
    print(f"Test set shape: {X_test.shape}")
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    # Print results
    print_results(results)
    
    # Save confusion matrix
    plot_confusion_matrix(results['confusion_matrix'])
    
    # Save results to JSON
    results_file = 'inference_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1'])
        }, f, indent=4)
    
    print(f"Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    main()

