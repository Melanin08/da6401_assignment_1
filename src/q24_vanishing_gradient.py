import argparse
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot


def parse_arguments():
    parser = argparse.ArgumentParser(description="Question 2.4: Vanishing Gradient Analysis")
    parser.add_argument("-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_val_split(X, y, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split = int((1.0 - val_ratio) * X.shape[0])
    train_idx = indices[:split]
    val_idx = indices[split:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def get_batches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def accuracy_score_numpy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def run_experiment(args, activation, hidden_size):
    run_name = f"q24_{activation}_{'_'.join(map(str, hidden_size))}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "rmsprop",
            "activation": activation,
            "hidden_size": hidden_size,
            "loss": "cross_entropy",
            "weight_init": "xavier",
        }
    )

    class Args:
        pass

    model_args = Args()
    model_args.dataset = args.dataset
    model_args.hidden_size = hidden_size
    model_args.activation = activation
    model_args.loss = "cross_entropy"
    model_args.weight_init = "xavier"
    model_args.optimizer = "rmsprop"
    model_args.learning_rate = args.learning_rate
    model_args.weight_decay = 0.0

    X_train_full, y_train_full, _, _ = load_data(args.dataset)
    X_train, y_train, X_val, y_val = train_val_split(X_train_full, y_train_full, seed=args.seed)

    y_train_oh = one_hot(y_train, 10)
    y_val_oh = one_hot(y_val, 10)

    model = NeuralNetwork(model_args)

    for epoch in range(args.epochs):
        batch_losses = []

        for X_batch, y_batch in get_batches(X_train, y_train_oh, args.batch_size):
            logits = model.forward(X_batch)

            from ann.objective_functions import cross_entropy_loss
            loss, _ = cross_entropy_loss(logits, y_batch)
            batch_losses.append(loss)

            model.backward(y_batch, None)

            first_layer_grad_norm = np.linalg.norm(model.layers[0].grad_W)

            model.update_weights()

            wandb.log({
                "epoch": epoch + 1,
                "batch_loss": float(loss),
                "first_layer_grad_norm": float(first_layer_grad_norm),
            })

        train_eval = model.evaluate(X_train, y_train_oh)
        val_eval = model.evaluate(X_val, y_val_oh)

        train_pred = train_eval["predictions"]
        val_pred = val_eval["predictions"]

        train_acc = accuracy_score_numpy(y_train, train_pred)
        val_acc = accuracy_score_numpy(y_val, val_pred)

        wandb.log({
            "epoch_summary": epoch + 1,
            "train_loss": train_eval["loss"],
            "val_loss": val_eval["loss"],
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "epoch_loss_mean": float(np.mean(batch_losses)),
        })

        print(
            f"{run_name} | Epoch {epoch + 1}/{args.epochs} | "
            f"Train Acc: {train_acc:.6f} | Val Acc: {val_acc:.6f}"
        )

    wandb.finish()


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    configs = [
        [64, 64],
        [128, 128],
        [128, 128, 128],
    ]

    for hidden_size in configs:
        run_experiment(args, activation="sigmoid", hidden_size=hidden_size)
        run_experiment(args, activation="relu", hidden_size=hidden_size)


if __name__ == "__main__":
    main()