"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
import argparse
import numpy as np

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument("--dataset", type=str, default="mnist",choices=["mnist", "fashion_mnist"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="sgd",choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, nargs="+", default=[128,128])
    parser.add_argument("--activation", type=str, default="relu",choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--loss", type=str, default="cross_entropy",choices=["cross_entropy", "mse"])
    parser.add_argument("--weight_init", type=str, default="xavier",choices=["xavier", "random"])
    parser.add_argument("--wandb_project", type=str, default="dl_assignment")
    parser.add_argument("--model_save_path", type=str, default="src/best_model")
    return parser.parse_args()

def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data
    

import os

def main():
    args = parse_arguments()

    dataset = getattr(args, "dataset", "mnist")
    loaded_data = load_data(dataset)

    epochs = getattr(args, "epochs", 10)
    batch_size = getattr(args, "batch_size", 32)

    model_save_path = getattr(args, "model_save_path", "src/best_model.npy")

    X_train, y_train, X_test, y_test = loaded_data

    model = NeuralNetwork(args)

    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

    best_weights = model.get_weights()

    if not model_save_path.endswith(".npy"):
        model_save_path += ".npy"

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    np.save(model_save_path, best_weights)

    print("Training complete")



if __name__ == '__main__':
    main()
