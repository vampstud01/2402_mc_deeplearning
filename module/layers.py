import numpy as np

class Sigmoid:
    def __init__(self):
        self.y = None
        
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y
    
    def backward(self, dout=1):
        dx = dout * self.y * (1 - self.y)
        return dx
    

class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, Z):
        self.mask = Z > 0
        return np.where(self.mask, Z, 0)
    
    def backward(self, dout):
        return dout * self.mask
    

class 완전연결:
    def __init__(self, 입력, 출력):
        self.W = np.random.randn(입력, 출력)
        self.b = np.zeros(출력)
        self.X = None
        self.dW = None
        self.db = None
        
    def forward(self, X):
        self.X = X
        Z = np.dot(X, self.W) + self.b
        return Z
    
    def backward(self, dout=1):
        dX = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        return dX
