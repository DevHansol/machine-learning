import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def identity_function(x):
    return x

def init_network():
    network = {} # 딕셔너리인데 안에 벡터를 저장함

    for i in range(1, 4):
        if i == 1:
            network[f'w{i}'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        elif i == 2:
            network[f'w{i}'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        else:
            network[f'w{i}'] = np.array([[0.1, 0.3], [0.2, 0.4]])

    for i in range(1, 4):
        if i == 1:
            network[f'b{i}'] = np.array([0.1, 0.2, 0.3])
        elif i == 2:
            network[f'b{i}'] = np.array([0.1, 0.2])
        else:
            network[f'b{i}'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # for i in range(1, 4):
    #     f'{network[f"w{i}"]}'
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b2
    y = identity_function(a3)

    return y

if __name__ == '__main__':
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)