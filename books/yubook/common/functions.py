import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)

def cross_entropy_error(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, y_true.size)
        y_pred = y_pred.reshape(1, y_pred.size)

    if y_pred.size == y_true.size:
        y_pred = y_pred.argmax(axis=1)

    batch_size = y_true.shape[0]
    return -np.sum(np.log(y_true[np.arange(batch_size), y_pred] + 1e-7)) / batch_size