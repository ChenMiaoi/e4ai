import pickle
import numpy as np
from datasets import mnist
from common.functions import sigmoid, softmax

def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_mnist(normalize=True, flatten=True,
                                                            one_hot_label=False)
    return x_test, y_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()
batch_size = 100
accuracy = 0
for epoch in range(0, len(x), batch_size):
    y_pred = predict(network, x[epoch:epoch + batch_size])
    print("epoch:", epoch, "accuracy:", accuracy / len(x), "%",
          "y_pred:", y_pred, "tets:", t[epoch:epoch + batch_size])
    p = np.argmax(y_pred, axis=1)
    accuracy += np.sum(p == t[epoch:epoch + batch_size])

print("Accuracy: ", str(float(accuracy) / len(x)))