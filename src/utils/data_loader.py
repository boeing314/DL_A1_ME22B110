import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist

def load_data(dataset="mnist"):

    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize pixels in the range 0 to 1
    X_train=X_train/255.0
    X_test=X_test/255.0

    return X_train, y_train, X_test, y_test