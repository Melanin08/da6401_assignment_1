import argparse
import wandb

from train import train


def sweep_train():
    wandb.init(settings=wandb.Settings(init_timeout=240))
    config = wandb.config

    class Args:
        pass

    args = Args()
    args.dataset = config.dataset
    args.epochs = config.epochs
    args.batch_size = config.batch_size
    args.learning_rate = config.learning_rate
    args.optimizer = config.optimizer
    args.weight_decay = config.weight_decay
    args.num_layers = len(config.hidden_size)
    args.hidden_size = list(config.hidden_size)
    args.activation = config.activation
    args.loss = config.loss
    args.weight_init = config.weight_init
    args.wandb_project = "da6401-assignment1"
    args.model_save_path = "src/best_model.npy"
    args.config_save_path = "src/best_config.json"
    args.seed = 42

    train(args, init_wandb=False)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Question 2.2: Hyperparameter Sweep")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    sweep_config = {
        "name": "q22_hyperparameter_sweep",
        "method": "random",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "dataset": {"values": ["mnist"]},
            "epochs": {"values": [5, 10]},
            "batch_size": {"values": [32, 64, 128]},
            "learning_rate": {"values": [0.0001, 0.0005, 0.001, 0.01]},
            "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop"]},
            "weight_decay": {"values": [0.0, 0.0001]},
            "hidden_size": {
                "values": [
                    [64],
                    [128, 128],
                    [128, 128, 128]
                ]
            },
            "activation": {"values": ["relu", "tanh", "sigmoid"]},
            "loss": {"values": ["cross_entropy"]},
            "weight_init": {"values": ["random", "xavier"]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="da6401-assignment1")
    print("Sweep created:", sweep_id)

    wandb.agent(sweep_id, function=sweep_train, count=args.count)


if __name__ == "__main__":
    main()