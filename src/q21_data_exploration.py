import argparse
import wandb

from utils.data_loader import load_data, get_class_names


def parse_arguments():
    parser = argparse.ArgumentParser(description="Question 2.1: Data Exploration and Class Distribution")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices=["mnist", "fashion_mnist"])
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment1")
    return parser.parse_args()


def log_sample_images_table(X, y, dataset_name):
    class_names = get_class_names(dataset_name)

    table = wandb.Table(columns=["class_id", "class_name", "sample_number", "image"])

    counts = {i: 0 for i in range(10)}

    for i in range(X.shape[0]):
        label = int(y[i])

        if counts[label] < 5:
            image = X[i].reshape(28, 28)
            table.add_data(
                label,
                class_names[label],
                counts[label] + 1,
                wandb.Image(image)
            )
            counts[label] += 1

        if all(counts[c] >= 5 for c in range(10)):
            break

    wandb.log({"sample_images_by_class": table})


def main():
    args = parse_arguments()

    wandb.init(
        project=args.wandb_project,
        name=f"q21_{args.dataset}_data_exploration",
        config={"dataset": args.dataset}
    )

    X_train, y_train, _, _ = load_data(args.dataset)

    log_sample_images_table(X_train, y_train, args.dataset)

    wandb.finish()
    print("Question 2.1 completed successfully.")


if __name__ == "__main__":
    main()