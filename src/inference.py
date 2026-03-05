import numpy as np
import argparse
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate trained model")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--model_path", type=str, default="best_model.npy")

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, nargs="+", default=[128,128])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--weight_init", type=str, default="xavier")

    return parser.parse_args()


def load_model(model_path):
    """
    Load model weights from .npy file
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def main():

    args = parse_arguments()

    # Load dataset
    X_train, y_train, X_test, y_test = load_data(args.dataset)

    # Build model
    model = NeuralNetwork(args)

    # Load weights
    weights = load_model(args.model_path)
    model.set_weights(weights)

    # Forward pass
    logits = model.forward(X_test)

    # Predictions
    y_pred = np.argmax(logits, axis=1)

    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print("Accuracy :", acc)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


if __name__ == "__main__":
    main()