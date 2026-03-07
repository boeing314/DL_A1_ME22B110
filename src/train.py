"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
import argparse
import numpy as np
import os
import wandb

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
    parser.add_argument("-d","--dataset", type=str, default="mnist",choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e","--epochs", type=int, default=10)
    parser.add_argument("-b","--batch_size", type=int, default=32)
    parser.add_argument("-l","--loss", type=str, default="cross_entropy",choices=["cross_entropy", "mse"])
    parser.add_argument("-o","--optimizer", type=str, default="sgd",choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr","--learning_rate", type=float, default=0.01)
    parser.add_argument("-wd","--weight_decay", type=float, default=0)
    parser.add_argument("-nhl","--num_layers", type=int, default=2)
    parser.add_argument("-sz","--hidden_size", nargs="+", default=["128","128"])
    parser.add_argument("-a","--activation", type=str, default="relu",choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-w_i","--weight_init", type=str, default="xavier",choices=["xavier", "random","zero"])
    parser.add_argument("-w_p","--wandb_project", type=str, default="test_project")
    parser.add_argument("-w_rn","--wandb_run_name", type=str, default="run_test")
    parser.add_argument("-msp","--model_save_path", type=str, default="src/best_model.npy")
    
    return parser.parse_args()

def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data
    



def main():
    args = parse_arguments()
    print('yes')
    print(args.hidden_size)
    if len(args.hidden_size) == 1 and "," in args.hidden_size[0]:
        args.hidden_size = [int(x) for x in args.hidden_size[0].split(",")]
    else:
        args.hidden_size = [int(x) for x in args.hidden_size]
    args.num_layers = len(args.hidden_size)
    wandb.init(project=args.wandb_project, name=args.wandb_run_name,config=vars(args))
    dataset = getattr(args, "dataset")
    model_save_path = getattr(args, "model_save_path", "sample_model.npy")

    X_train, y_train, X_val, y_val = load_data(dataset)

    model = NeuralNetwork(args)

    model.train(X_train, y_train)

    val_metrics = model.evaluate(X_val, y_val)

    
    wandb.log({"val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"], "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"], "val_f1": val_metrics["f1"]})

    weights = model.get_weights()

    if not model_save_path.endswith(".npy"):
        model_save_path += ".npy"
    dir_path = os.path.dirname(model_save_path)
    if dir_path != "":
        os.makedirs(dir_path, exist_ok=True)
    np.save(model_save_path, weights)

    print("Training complete")



if __name__ == '__main__':
    main()