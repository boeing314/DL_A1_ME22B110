import numpy as np
import argparse
import ast
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_arguments():

    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument("-d","--dataset", type=str, default="mnist",choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e","--epochs", type=int, default=10)
    parser.add_argument("-b","--batch_size", type=int, default=32)
    parser.add_argument("-l","--loss", type=str, default="cross_entropy",choices=["cross_entropy", "mse"])
    parser.add_argument("-o","--optimizer", type=str, default="sgd",choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr","--learning_rate", type=float, default=0.01)
    parser.add_argument("-wd","--weight_decay", type=float, default=0)
    parser.add_argument("-nhl","--num_layers", type=int, default=4)
    parser.add_argument("-sz","--hidden_size", type=int, nargs="+", default=[128,128,128,128])
    parser.add_argument("-a","--activation", type=str, default="relu",choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-w_i","--weight_init", type=str, default="xavier",choices=["xavier", "random","zero"])
    parser.add_argument("-w_p","--wandb_project", type=str, default="test_project")
    parser.add_argument("-w_rn","--wandb_run_name", type=str, default="run_test")

    

    return parser.parse_args()


def load_model(model_path):
    weights = np.load(model_path, allow_pickle=True).item()
    return weights

def evaluate_model(model, X_test, y_test): 
    metrics = model.evaluate(X_test, y_test)
    return metrics


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    dataset = getattr(args, "dataset")
    X_train, y_train, X_test, y_test = load_data(dataset)
    model = NeuralNetwork(args)
    weights = load_model(args.model_load_path)
    model.set_weights(weights)
    result = evaluate_model(model, X_test, y_test)
    print("Evaluation complete!")
    print(f"Logits: {result['logits']}")
    print(f"Loss: {result['loss']:.4f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")

    return result


if __name__ == '__main__':
    main()