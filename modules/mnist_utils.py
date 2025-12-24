import struct

def load_mnist_images(path, limit=None):
    with open(path, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = []
        for _ in range(num if limit is None else limit):
            img = list(f.read(rows * cols))
            images.append([p / 255.0 for p in img])
        return images

def load_mnist_labels(path, limit=None):
    with open(path, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return list(f.read(num if limit is None else limit))

def one_hot(labels, C=10):
    y = []
    for l in labels:
        row = [0]*C
        row[l] = 1
        y.append(row)
    return y
