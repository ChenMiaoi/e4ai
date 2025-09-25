import numpy as np
from matplotlib import pyplot as plt

from books.yubook.datasets import mnist
from books.yubook.project2.two_layer_net import TwoLayerNet

(x_train, y_train), (x_test, y_test) = mnist.load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = y_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_accuracy = network.accuracy(x_train, y_train)
        test_accuracy = network.accuracy(x_test, y_test)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        print("train accuracy: {}, test accuracy: {}".format(train_accuracy, test_accuracy))


markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_accuracy_list))
plt.plot(x, train_accuracy_list, marker=markers['train'], markersize=8, label='train')
plt.plot(x, test_accuracy_list, marker=markers['test'], markersize=8, label='test', linestyle='--')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()