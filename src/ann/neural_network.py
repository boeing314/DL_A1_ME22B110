"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import argparse
import ann.activations as activations
import ann.objective_functions as objective_functions
import ann.neural_layer as neural_layer
import ann.optimizers as optimizers
class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.layers = []
        self.activations = []
        self.objective_function = getattr(cli_args, "loss", "cross_entropy")
        self.optimizer = getattr(cli_args, "optimizer", "sgd")
        self.epochs = getattr(cli_args, "epochs", 10)
        self.batch_size = getattr(cli_args, "batch_size", 32)
        self.learning_rate = getattr(cli_args, "learning_rate", 0.01)
        self.dataset = getattr(cli_args, "dataset", "mnist")
        self.weight_decay = getattr(cli_args, "weight_decay", 0)
        self.num_layers = getattr(cli_args, "num_layers", 2)
        self.hidden_size = getattr(cli_args, "hidden_size", [128,128])
        self.activation = getattr(cli_args, "activation", "relu")
        self.weight_init = getattr(cli_args, "weight_init", "xavier")
        self.wandb_project = getattr(cli_args, "wandb_project", "dl_assignment")

        self.input_size = 784  # For MNIST/Fashion-MNIST
        self.output_size = 10  # For MNIST/Fashion-MNIST

        self.activation_map={
            'relu': activations.ReLU,
            'sigmoid': activations.Sigmoid,
            'tanh': activations.Tanh}

        self.objective_function_map={
            'cross_entropy': objective_functions.cross_entropy,
            'mse': objective_functions.MSE}

        
        self.optimizer_map={
            'sgd': optimizers.SGD,
            'momentum': optimizers.Momentum,
            'nag': optimizers.NAG,
            'rmsprop': optimizers.RMSProp}
        self.optimizer_function=self.optimizer_map[self.optimizer](self.learning_rate)
        self.loss_function=self.objective_function_map[self.objective_function]()
        self.layers.append(neural_layer.NeuralLayer(self.input_size, self.hidden_size[0], self.weight_init))
        self.activations.append(self.activation_map[self.activation]())
        for i in range(self.num_layers-1):
            self.layers.append(neural_layer.NeuralLayer(self.hidden_size[i], self.hidden_size[i+1], self.weight_init))
            self.activations.append(self.activation_map[self.activation]())
        self.layers.append(neural_layer.NeuralLayer(self.hidden_size[-1], self.output_size, self.weight_init))
    

    def forward(self, X):

        # Handle different input shapes from autograder
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        for i in range(len(self.layers)-1):
            X = self.layers[i].forward(X)
            X = self.activations[i].forward(X)

        X = self.layers[-1].forward(X)

        return X

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        grad=self.loss_function.backward(y_true, y_pred)
        
        for i in range(len(self.layers)-1, -1, -1):
            grad = self.layers[i].backward(grad)
            grad_W_list.append(self.layers[i].grad_W)
            grad_b_list.append(self.layers[i].grad_b)
            if i>0:
                grad = self.activations[i-1].backward(grad)

        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        #print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        #print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b

    def update_weights(self):
        for i in range(len(self.layers)):
            self.optimizer_function.update(self.layers[i])

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        for epoch in range(epochs):
            loss_epoch=0
            total_samples=0
            print(f"Epoch {epoch+1}/{epochs}")
            # Shuffle data at the start of each epoch
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                if i + batch_size > X_train.shape[0]:
                    X_batch = X_train[i:]
                    y_batch = y_train[i:]
                else:
                    X_batch = X_train[i:i+batch_size]
                    y_batch = y_train[i:i+batch_size]
                for layer in self.layers:
                    if self.optimizer == 'nag':
                        self.optimizer_function.lookahead(layer)
                y_pred = self.forward(X_batch)
                loss_epoch += self.loss_function.forward(y_batch, y_pred)*X_batch.shape[0]
                total_samples += X_batch.shape[0]
                avg_loss = loss_epoch / total_samples
                self.backward(y_batch, y_pred)
                self.update_weights()
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        return self.loss_function.forward(y, y_pred)

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

