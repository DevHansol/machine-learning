import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle # 용량을 작게 저장하는 모듈
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import matplotlib.pyplot as plt
import random

def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_train, y_train

if __name__ == '__main__':
    x_train, y_train = get_data()
    plt.figure()
    plt.imshow(x_train[0][0])
    plt.colorbar()
    plt.show()

    I = np.random.choice(60000, 25)

    plt.figure(figsize = (10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_train[I[i]][0], cmap = plt.cm.binary)
        plt.xlabel(y_train[I[i]])
    plt.show()