# coding: utf-8
import time
import numpy as np


class 역전파신경망:
    def __init__(self, 손실함수):
        self.layers = []
        self.loss_func = 손실함수
        
    def add(self, layer):
        self.layers.append(layer)
        
    def predict(self, X):
        """순전파 (feedforward)"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def 손실산출(self, X, y):
        z_last = self.predict(X)
        손실 = self.loss_func.forward(z_last, y)
        return 손실
    
    def fit(self, X, y, 배치크기, 에폭수, 최적화, 성능지표=None):
        """학습"""
        표본수 = X.shape[0]
        에폭당_배치개수 = 표본수 // 배치크기
        손실변화 = []
        for 에폭번호 in range(에폭수):
            start_time = time.time()
            학습횟수 = 에폭당_배치개수
            for i in range(학습횟수):       
                # 1. 미니배치
                배치색인 = np.random.choice(표본수, 배치크기)
                X_batch = X[배치색인]
                y_batch = y[배치색인]
                # 2. 경사 산출 (역전파)
                # 1) 순전파
                self.손실산출(X_batch, y_batch)
                # 2) 역전파
                dout = self.loss_func.backward()
                for layer in reversed(self.layers):
                    dout = layer.backward(dout)

                # 3. 매개변수 갱신 (경사 하강)
                params = []
                grads = []
                for layer in self.layers:
                    if hasattr(layer, 'W') and hasattr(layer, 'b'):
                        params.extend([layer.W, layer.b])
                        grads.extend([layer.dW, layer.db])
                        
                최적화.update(params, grads)
            end_time = time.time()
            # (선택적) 손실확인
            손실 = self.손실산출(X, y)
            손실변화.append(손실)
    
            # # 학습 성능 평가
            소요시간 = end_time - start_time
            print(f'학습 {에폭번호+1} ({에폭당_배치개수}/학습) ({소요시간 * 1000:.0f} ms)')
            if not 성능지표:
                print(f'\t손실: {손실:.3f}')
            else:
                output = self.predict(X)
                print(f'\t손실: {손실:.3f}\t성능: {성능지표(y, output):.4f}')
                
        return 손실변화