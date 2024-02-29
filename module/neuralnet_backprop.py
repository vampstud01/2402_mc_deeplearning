import numpy as np

def 교차엔트로피오차(y, y_pred):
    delta = 1e-7
    배치크기 = y.shape[0]
    return -np.sum(y * np.log(y_pred + delta)) / 배치크기

def softmax(z):
    if z.ndim == 1:
        z = z.reshape(1, -1)
    exp_z = np.exp(z - np.max(z, axis=1).reshape(-1, 1))
    return exp_z / np.sum(exp_z, axis=1).reshape(-1, 1)

class Sigmoid:
    def __init__(self):
        self.y = None
        
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y
    
    def backward(self, dout):
        dx = dout * self.y * (1 - self.y)
        return dx

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
    
    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        return dX

class SoftmaxCrossEntropy:
    def __init__(self):
        self.Y = None
        self.Y_pred = None
        
    def forward(self, Z, Y):
        self.Y = Y
        self.Y_pred = softmax(Z)
        손실 = 교차엔트로피오차(Y, self.Y_pred)
        return 손실
    
    def backward(self, dout=1):
        배치크기 = self.Y.shape[0]
        dZ = self.Y_pred - self.Y
        return dZ / 배치크기

class 역전파신경망:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def predict(self, X):
        output = X
        for layer in self.layers[:-1]:
            output = layer.forward(output)
        return output
    
    def 손실산출(self, X, y):
        Z_last = self.predict(X)
        마지막계층 = self.layers[-1]
        손실 = 마지막계층.forward(Z_last, y)
        return 손실
    
    def fit(self, X, y, 배치크기, 학습횟수, 학습률):
        표본수 = X.shape[0]
        손실변화 = []
        for i in range(학습횟수):
            # 1. 미니배치
            배치색인 = np.random.choice(표본수, 배치크기)
            X_batch = X[배치색인]
            y_batch = y[배치색인]
            # 2. 경사산출 (오차역전파)
            # 1) 순전파
            self.손실산출(X_batch, y_batch)
            # 2) 역전파
            dout = 1
            for layer in reversed(self.layers):
                dout = layer.backward(dout)
            # 3. 매개변수 갱신 (경사하강)
            for layer in self.layers:
                if isinstance(layer, 완전연결):
                    layer.W -= layer.dW * 학습률
                    layer.b -= layer.db * 학습률
                    
            # 손실 확인
            손실 = self.손실산출(X_batch, y_batch)
            손실변화.append(손실)
            if i == 0 or (i + 1) % 100 == 0:
                print(f'학습 {i+1}')
                print(f'\t손실: {손실}')
        return 손실변화