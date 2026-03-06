"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
class NeuralLayer:
    def __init__(self,input_size,output_size,weight_init='xavier'):
        self.input_size=input_size
        self.output_size=output_size
        self.variance=2/(input_size+output_size)
        if weight_init=='xavier':
            self.W=np.random.randn(input_size,output_size)*np.sqrt(self.variance)
        else:
            self.W=np.random.randn(input_size,output_size)*0.01
        self.b=np.zeros((1,output_size))
    def forward(self,x):
        self.x=x
        return x@self.W+self.b
    def backward(self,grad_output):
        self.grad_W=(self.x.T@grad_output)
        self.grad_b=np.sum(grad_output,axis=0)
        return grad_output@self.W.T