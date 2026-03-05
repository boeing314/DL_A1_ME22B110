"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np
class ReLU:
    def forward(self,x):
        self.x=x
        return np.maximum(0,x)
    def backward(self,grad_input):
        return grad_input*(self.x>0)

class Sigmoid:
    def forward(self,x):
        self.out=1/(1+np.exp(-x))
        return self.out
    def backward(self,grad_input):
        return grad_input*self.out*(1-self.out)

class Tanh:
    def forward(self,x):
        self.out=np.tanh(x)
        return self.out
    def backward(self,grad_input):
        return grad_input*(1-self.out**2)
