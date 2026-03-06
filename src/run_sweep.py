"""
Hyperparameter Sweep Script
Uses W&B Sweeps for systematic hyperparameter search
"""
import wandb
from train import main as train_main
import argparse

def sweep_agent():
    """Run sweep agent"""
    # Initialize wandb with sweep config
    wandb.init()

    # Convert wandb config to argparse format
    args = argparse.Namespace()

    # Map wandb config to args
    config = wandb.config
    args.dataset = config.dataset
    args.epochs = config.epochs
    args.batch_size = config.batch_size
    args.learning_rate = config.learning_rate
    args.weight_decay = config.weight_decay
    args.optimizer = config.optimizer
    args.num_layers = config.num_layers
    args.hidden_size = config.hidden_size
    args.activation = config.activation
    args.loss = config.loss
    args.weight_init = config.weight_init
    args.wandb_project = "da6401-assignment"
    args.model_save_path = f"models/sweep_model_{wandb.run.name}.npy"

    # Run training
    try:
        train_main(args)
    except Exception as e:
        print(f"Sweep run failed: {e}")
        wandb.log({"sweep_error": str(e)})

if __name__ == "__main__":
    # Define sweep configuration
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'dataset': {'values': ['mnist']},
            'epochs': {'values': [5, 10]},
            'batch_size': {'values': [32, 64, 128]},
            'learning_rate': {'values': [0.001, 0.01, 0.1]},
            'weight_decay': {'values': [0.0, 0.0001]},
            'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop']},
            'num_layers': {'values': [1, 2, 3]},
            'hidden_size': {'values': [[64], [128, 64], [128, 128, 64]]},
            'activation': {'values': ['relu', 'sigmoid', 'tanh']},
            'loss': {'values': ['cross_entropy']},
            'weight_init': {'values': ['random', 'xavier']}
        }
    }

    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="da6401-assignment")

    print(f"Sweep created: {sweep_id}")
    print("Run the following command to start the sweep:")
    print(f"wandb agent {sweep_id}")

