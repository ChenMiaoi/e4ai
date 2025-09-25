import os
import sys

import numpy as np
from PIL import Image

from datasets.mnist import init_mnist

sys.path.append(os.pardir)
from datasets import mnist

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

init_mnist()
(x_train, y_train), (x_test, y_test) = mnist.load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = y_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)