"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np
class SGD():
    def __init__(self,lr=0.01,weight_decay=0):
        self.lr=lr
        self.weight_decay=weight_decay
    def update(self,layer):
        layer.W-=self.lr*(layer.grad_W+self.weight_decay*layer.W)
        layer.b-=self.lr*(layer.grad_b+self.weight_decay*layer.b)

class Momentum():
    def __init__(self,lr=0.01,momentum=0.9,weight_decay=0):
        self.lr=lr
        self.momentum=momentum
        self.weight_decay=weight_decay
        self.v_W={}
        self.v_b={}
    def update(self,layer):
        layer_id=id(layer)
        if layer_id not in self.v_W:
            self.v_W[layer_id]=np.zeros_like(layer.W)
            self.v_b[layer_id]=np.zeros_like(layer.b)
        self.v_W[layer_id]=self.momentum*self.v_W[layer_id]+self.lr*layer.grad_W+self.lr*self.weight_decay*layer.W
        self.v_b[layer_id]=self.momentum*self.v_b[layer_id]+self.lr*layer.grad_b+self.lr*self.weight_decay*layer.b
        layer.W-=self.v_W[layer_id]
        layer.b-=self.v_b[layer_id]

class NAG():
    def __init__(self,lr=0.01,weight_decay=0,momentum=0.9):
        self.lr=lr
        self.weight_decay=weight_decay
        self.momentum=momentum
        self.v_W={}
        self.v_b={}
    def lookahead(self,layer):
        layer_id=id(layer)
        self.ori_W=layer.W.copy()
        self.ori_b=layer.b.copy()
        if layer_id not in self.v_W:
            self.v_W[layer_id]=np.zeros_like(layer.W)
            self.v_b[layer_id]=np.zeros_like(layer.b)
        layer.W-=self.momentum*self.v_W[layer_id]
        layer.b-=self.momentum*self.v_b[layer_id]
    def update(self,layer):
        layer_id=id(layer)
        if layer_id not in self.v_W:
            self.v_W[layer_id]=np.zeros_like(layer.W)
            self.v_b[layer_id]=np.zeros_like(layer.b)
        self.v_W[layer_id]=self.momentum*self.v_W[layer_id]+self.lr*layer.grad_W+self.lr*self.weight_decay*self.ori_W
        self.v_b[layer_id]=self.momentum*self.v_b[layer_id]+self.lr*layer.grad_b+self.lr*self.weight_decay*self.ori_b
        layer.W=self.ori_W-self.v_W[layer_id]
        layer.b=self.ori_b-self.v_b[layer_id]

class RMSProp():
    def __init__(self,lr=0.001,decay=0.9,weight_decay=0,eps=1e-8):
        self.lr=lr
        self.decay=decay
        self.weight_decay=weight_decay
        self.eps=eps
        self.s_W={}
        self.s_b={}
    def update(self,layer):
        layer_id=id(layer)
        if layer_id not in self.s_W:
            self.s_W[layer_id]=np.zeros_like(layer.W)
            self.s_b[layer_id]=np.zeros_like(layer.b)
        self.s_W[layer_id]=self.decay*self.s_W[layer_id]+(1-self.decay)*(layer.grad_W+self.weight_decay*layer.W)**2
        self.s_b[layer_id]=self.decay*self.s_b[layer_id]+(1-self.decay)*(layer.grad_b+self.weight_decay*layer.b)**2
        layer.W-=0
        layer.b-=0