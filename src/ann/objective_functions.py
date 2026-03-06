"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

class MSE:
    def forward(self,y_true,y_pred):
        return np.mean((y_true-y_pred)**2)
    def backward(self,y_true,y_pred):
        N=y_true.shape[1]
        B=y_true.shape[0]
        return 2*(y_pred-y_true)/(B*N)

import numpy as np

class cross_entropy:

    def forward(self, y_true, y_pred):
        """
        y_pred : logits (B, C)
        y_true : labels (B,) or one-hot (B, C)
        """

        # Convert integer labels to one-hot if needed
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        # Numerical stability trick
        shifted_logits = y_pred - np.max(y_pred, axis=1, keepdims=True)

        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.y_true = y_true

        B = y_true.shape[0]

        loss = -np.sum(y_true * np.log(self.probs + 1e-12)) / B

        return loss


    def backward(self, y_true, y_pred):
        """
        Gradient of loss w.r.t logits
        """

        # Convert integer labels to one-hot if needed
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]
        shifted_logits = y_pred - np.max(y_pred, axis=1, keepdims=True)

        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.y_true = y_true

        B = y_true.shape[0]

        grad = (self.probs - y_true) / B

        return grad
