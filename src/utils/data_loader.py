import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist

def one_hot(y, num_classes=10):
    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot


def load_data(dataset="mnist", val_split=0.1):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize pixels
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # One-hot encode labels
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    return X_train,y_train,X_test,y_test