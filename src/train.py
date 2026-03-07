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
    parser.add_argument("-e","--epochs", type=int, default=30)
    parser.add_argument("-b","--batch_size", type=int, default=32)
    parser.add_argument("-l","--loss", type=str, default="cross_entropy",choices=["cross_entropy", "mse"])
    parser.add_argument("-o","--optimizer", type=str, default="sgd",choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-lr","--learning_rate", type=float, default=0.05)
    parser.add_argument("-wd","--weight_decay", type=float, default=0)
    parser.add_argument("-nhl","--num_layers", type=int, default=2)
    parser.add_argument("-sz","--hidden_size", nargs="+", default=["128","128"])
    parser.add_argument("-a","--activation", type=str, default="relu",choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-w_i","--weight_init", type=str, default="xavier",choices=["xavier", "random","zero"])
    parser.add_argument("-w_p","--wandb_project", type=str, default="dl_a1")
    parser.add_argument("-w_rn","--wandb_run_name", type=str, default="run_test")
    parser.add_argument("-msp","--model_save_path", type=str, default="src/test_model.npy")
    
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
    wandb.init(project=args.wandb_project, group='test_group', name=args.wandb_run_name,config=vars(args))
    dataset = getattr(args, "dataset")
    model_save_path = getattr(args, "model_save_path", "sample_model.npy")

    X_train, y_train, X_val, y_val = load_data(dataset)
    '''
    args.epochs = 1
    model = NeuralNetwork(args)
    for epoch in range(20):
        
        model.train(X_train, y_train)
        val_metrics = model.evaluate(X_val, y_val)
        val_loss= val_metrics["loss"]
        print(val_loss)
        wandb.log({"val_loss": val_loss,"epoch": epoch+1})
    train_metrics = model.evaluate(X_train, y_train)

    '''
    model = NeuralNetwork(args)
    model.train(X_train, y_train)
    wandb.log({"val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"], "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"], "val_f1": val_metrics["f1"],"train_accuracy": train_metrics["accuracy"],"train_f1": train_metrics["f1"],"train_loss": train_metrics["loss"]})

    weights = model.get_weights()

    if not model_save_path.endswith(".npy"):
        model_save_path += ".npy"
    dir_path = os.path.dirname(model_save_path)
    if dir_path != "":
        os.makedirs(dir_path, exist_ok=True)
    np.save(model_save_path, weights)

    print("Training complete")
'''
    table = wandb.Table(columns=["class", "image"])

    for i in range(10):
        idxs = np.where(y_train == i)[0][:5]
        for idx in idxs:
            img = X_train[idx].reshape(28, 28)
            table.add_data(i, wandb.Image(img))
    wandb.log({"sample_images": table})
'''
'''
    for i in [2,4,6,8]:
        args.num_layers = i
        args.hidden_size = [128]*(i+1)
        model = NeuralNetwork(args)
        model.train(X_train, y_train)
'''

if __name__ == '__main__':
    main()