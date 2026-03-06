import argparse
import numpy as np
import json
import os
import wandb
from sklearn.model_selection import train_test_split
from utils.data_loader import load_data, one_hot
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from ann.objective_functions import get_loss_function


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Training Multi-layer Perceptron for image classification"
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        choices=["mnist", "fashion_mnist"],
        help="Dataset to use"
    )

    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )

    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate"
    )

    parser.add_argument(
        "-wd", "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization)"
    )

    parser.add_argument(
        "-o", "--optimizer",
        type=str,
        required=True,
        choices=["sgd", "momentum", "nag", "rmsprop"],
        help="Optimizer to use"
    )

    parser.add_argument(
        "-nhl", "--num_layers",
        type=int,
        default=2,
        help="Number of hidden layers"
    )

    parser.add_argument(
        "-sz", "--hidden_size",
        type=int,
        nargs="+",
        required=True,
        help="Number of neurons in each hidden layer"
    )

    parser.add_argument(
        "-a", "--activation",
        type=str,
        required=True,
        choices=["relu", "sigmoid", "tanh"],
        help="Activation function for hidden layers"
    )

    parser.add_argument(
        "-l", "--loss",
        type=str,
        required=True,
        choices=["cross_entropy", "mse"],
        help="Loss function to use"
    )

    parser.add_argument(
        "-w_i", "--weight_init",
        type=str,
        default="xavier",
        choices=["random", "xavier"],
        help="Weight initialization method"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="da6401-assignment",
        help="W&B project name"
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        default="best_model.npy",
        help="Path to save best model"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_arguments()

    # Initialize W&B
    wandb.init(project=args.wandb_project, config=vars(args))
     
    # Give the run a clear name
    wandb.run.name = f"{args.optimizer}_{args.activation}_lr{args.learning_rate}_bs{args.batch_size}_layers{'-'.join(map(str,args.hidden_size))}"

    print(f"Loading {args.dataset} dataset...")
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_data(args.dataset)
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    # One-hot encode labels
    y_train = one_hot(y_train)
    y_val = one_hot(y_val)
    y_test = one_hot(y_test)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Create model
    print(f"Creating model with {len(args.hidden_size)} hidden layers...")
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=args.hidden_size,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init,
        use_softmax=True
    )

    # Create optimizer
    optimizer = get_optimizer(
        args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Create loss function
    loss_fn = get_loss_function(args.loss)()

    print(f"Starting training with {args.optimizer} optimizer for {args.epochs} epochs...")

    # Training loop
    model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=optimizer,
        loss_fn=loss_fn,
        verbose=True,
        wandb_log=wandb.log
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_logits = model.forward(X_test)
    test_predictions = np.argmax(test_logits, axis=1)
    test_labels = np.argmax(y_test, axis=1)
    test_acc = np.mean(test_predictions == test_labels)

    print(f"Test Accuracy: {test_acc:.4f}")

    # Save best model
    weights = model.get_weights()
    np.save(args.model_save_path, weights, allow_pickle=True)
    
    # Save config
    config_path = os.path.splitext(args.model_save_path)[0] + "_config.json"
    with open(config_path, "w") as f:
        json.dump({
            **vars(args),
            'test_accuracy': float(test_acc),
            'hidden_layers': args.hidden_size
        }, f, indent=4)

    print(f"\nModel saved to {args.model_save_path}")
    print(f"Config saved to {config_path}")

    wandb.log({'test_accuracy': test_acc})
    wandb.finish()

    print("Training completed!")


if __name__ == "__main__":
    main()

