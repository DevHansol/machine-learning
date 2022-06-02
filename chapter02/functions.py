import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype = np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

if __name__ == '__main__':
    X = np.arange(-5.0, 5.0, 0.001)
    y = step_function(X)
    plt.plot(X, y)
    plt.ylim(-0.1, 1.1)
    plt.show()