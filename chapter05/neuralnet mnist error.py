import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle # 용량을 작게 저장하는 모듈
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
import matplotlib.pyplot as plt

def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_test, y_test

def init_network(): # 저자가 학습시킨 뉴럴 네트워크를 불러와서 딕셔너리를 리턴함
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)

        return y

if __name__ == '__main__':
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0
    error = []

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis = 1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
        error.append(p != t[i:i+batch_size])

    print(error)
    print('Accuracy:', str(float(accuracy_cnt)/len(x)))

    plt.figure(figsize = (10, 10))
    for i in range(100):
        plt.subplot(20, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x[error[i]].reshape(28, 28), cmap = plt.cm.binary)
        plt.xlabel(str(t[error[i]]) + "==>" + str(p[i])) # 원래 숫자는 이것인데 ==> 머신이 대답한 숫자는 이것이다
    plt.show()