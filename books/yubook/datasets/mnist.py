import gzip
import os
import pickle
import urllib.request

import numpy as np

url_base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
key_file = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz"
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = os.path.join(dataset_dir, "mnist.pkl")

img_size = 28 * 28

def _download(filename: str):
    filepath = os.path.join(dataset_dir, filename)
    if os.path.exists(filepath):
        return

    print("Downloading ", filename)
    urllib.request.urlretrieve(url_base + filename, filepath)
    print("Done")

def download_mnist():
    for v in key_file.values():
        if not os.path.exists(v):
            _download(v)

def _load_label(filename: str):
    filepath = os.path.join(dataset_dir, filename)

    with gzip.open(filepath, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels

def _load_img(filename: str):
    filepath = os.path.join(dataset_dir, filename)

    with gzip.open(filepath, "rb") as f:
        img = np.frombuffer(f.read(), np.uint8, offset=16)
    img = img.reshape(-1, img_size)

    return img

def _convert_numpy():
    dataset = {
        "train_img": _load_img(key_file["train_img"]),
        "train_label": _load_label(key_file["train_label"]),
        "test_img": _load_img(key_file["test_img"]),
        "test_label": _load_label(key_file["test_label"])
    }
    return dataset

def _change_one_hot_label(label):
    T = np.zeros((label.size, 10))
    for idx, row in enumerate(T):
        row[label[idx]] = 1

    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, "rb") as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32) / 255.0

    if one_hot_label:
        dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_one_hot_label(dataset["test_label"])

    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset["train_img"], dataset["train_label"]), (dataset["test_img"], dataset["test_label"])

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("Done")
