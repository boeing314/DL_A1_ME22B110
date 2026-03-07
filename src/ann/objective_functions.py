"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

class MSE:
    def forward(self,y_true,y_pred):
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]
        return np.mean((y_true-y_pred)**2)
    def backward(self,y_true,y_pred):
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]
        N=y_true.shape[1]
        B=y_true.shape[0]
        return 2*(y_pred-y_true)/(B)

class cross_entropy:

    def forward(self, y_true, y_pred):

        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        exp_logits_shifted = np.exp(y_pred-np.max(y_pred, axis=1, keepdims=True))
        self.probs = exp_logits_shifted / np.sum(exp_logits_shifted, axis=1, keepdims=True)
        self.y_true = y_true

        B = y_true.shape[0]

        loss = -np.sum(y_true * np.log(self.probs + 1e-12)) / B

        return loss


    def backward(self, y_true, y_pred):

        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        exp_logits_shifted = np.exp(y_pred-np.max(y_pred, axis=1, keepdims=True))
        self.probs = exp_logits_shifted / np.sum(exp_logits_shifted, axis=1, keepdims=True)
        self.y_true = y_true

        B = y_true.shape[0]

        grad = (self.probs - y_true) / B

        return grad
