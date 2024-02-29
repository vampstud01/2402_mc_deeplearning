import numpy as np

def softmax(z):
    if z.ndim == 1:
        z = z.reshape(1, -1)
    exp_z = np.exp(z - np.max(z, axis=1).reshape(-1, 1))
    return exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)

def 교차엔트로피오차(y, y_pred):
    delta = 1e-7
    배치크기 = y.shape[0]
    return -np.sum(y * np.log(y_pred + delta)) / 배치크기


class SoftmaxCrossEntropy:
    def __init__(self):
        self.Y_pred = None
        self.Y = None
    
    def forward(self, Z, Y):
        self.Y = Y
        예측확률 = softmax(Z)
        self.Y_pred = 예측확률
        손실 = 교차엔트로피오차(Y, 예측확률)
        return 손실
    
    def backward(self, dout=1):
        dZ = self.Y_pred - self.Y
        dZ /= len(self.Y)
        return dZ