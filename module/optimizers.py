import numpy as np


class SGD:
    def __init__(self, 학습률=0.01):
        self.학습률 = 학습률
        
    def update(self, params, 경사):
        for W, dW in zip(params, 경사):
            W -= dW * self.학습률
            
            
class Momentum:
    def __init__(self, 학습률=0.01, momentum=0.9):
        self.학습률 = 학습률
        self.momentum = momentum
        self.v = None
        
    def update(self, params, 경사):
        # 매개변수별 속도 초기화
        if self.v is None:
            self.v = []
            for W in params:
                self.v.append(np.zeros_like(W).astype('float32'))
        
        for W, dW, v in zip(params, 경사, self.v):
            # 식 6.3
            v *= self.momentum
            v -= self.학습률 * dW
            # 식 6.4
            W += v