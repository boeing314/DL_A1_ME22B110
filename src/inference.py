import numpy as np
import argparse
import ast
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_arguments():

    parser = argparse.ArgumentParser(description="Evaluate trained model")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--model_path", type=str, default="src/best_model.npy")

    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--hidden_size", type=str, default="[128,128]")

    parser.add_argument("--activation", type=str, default="relu")

    parser.add_argument("--loss", type=str, default="cross_entropy")

    parser.add_argument("--weight_init", type=str, default="xavier")

    args = parser.parse_args()

    # Convert hidden_size string → list
    args.hidden_size = ast.literal_eval(args.hidden_size)

    # Ensure layer count matches hidden sizes
    args.num_layers = len(args.hidden_size)

    return args


def load_model(model_path):

    weights = np.load(model_path, allow_pickle=True).item()

    return weights


def main():

    args = parse_arguments()

    # Load dataset
    X_train, y_train, X_test, y_test = load_data(args.dataset)

    # Build model
    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)
    model = NeuralNetwork(args)

    # Load trained weights
    weights = load_model(args.model_path)
    model.set_weights(weights)

    # Forward pass
    logits = model.forward(X_test)

    # Predicted labels
    y_pred = np.argmax(logits, axis=1)

    # Handle one-hot vs integer labels
    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    # Metrics
    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    print("Accuracy :", acc)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


if __name__ == "__main__":
    main()