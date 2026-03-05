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

class cross_entropy:
    def forward(self,y_true,y_pred):
        # We implement softmax and get the probability corresponding to each digit
        exp_logits=np.exp(y_pred-np.max(y_pred,axis=1,keepdims=True))
        self.probs=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)
        # We implement a categorical cross entropy loss as the classes are mutually exclusive
        B=y_true.shape[0]
        return -np.sum(y_true*np.log(self.probs+1e-12))/B
    def backward(self,y_true,y_pred):
        exp_logits=np.exp(y_pred-np.max(y_pred,axis=1,keepdims=True))
        self.probs=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)
        B=y_true.shape[0]
        return (self.probs-y_true)/B
