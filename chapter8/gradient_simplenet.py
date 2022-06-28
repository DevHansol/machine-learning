import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

if __name__ == '__main__':
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    net = simpleNet()

    f = lambda w: net.loss(x, t)
    a = np.array([[1,2,3], [0,0,0]])
    net.W = a
    print(f(a))
    b = np.array([[0,0,0], [1,2,3]])
    net.W = b
    print(f(b))
    dW = numerical_gradient(f, net.W)

    print(dW)