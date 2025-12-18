import math
import random
import struct
import time

# =====================================================
# MNIST LOADER (NO NUMPY)
# =====================================================

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

# =====================================================
# MATRIX OPS (NO NUMPY)
# =====================================================

def mat_mul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]

def mat_add_bias(A, b):
    return [[A[i][j] + b[0][j] for j in range(len(b[0]))]
            for i in range(len(A))]

def transpose(A):
    return list(map(list, zip(*A)))

def subtract_matrix(A, B):
    return [[A[i][j]-B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]

def tanh_matrix(A):
    return [[math.tanh(x) for x in row] for row in A]

def tanh_derivative_matrix(A):
    return [[1-x*x for x in row] for row in A]

# =====================================================
# SOFTMAX
# =====================================================

def softmax(row):
    exps = [math.exp(x-max(row)) for x in row]
    s = sum(exps)
    return [e/s for e in exps]

def softmax_matrix(A):
    return [softmax(row) for row in A]

def argmax_row(row):
    return max(range(len(row)), key=lambda i: row[i])

# =====================================================
# FLEXIBLE ANN (FIXED)
# =====================================================

class FlexibleANN:
    def __init__(self, layer_sizes, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.layers = []
        self.vel = []

        for i in range(len(layer_sizes)-1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]

            # Xavier initialization
            limit = math.sqrt(6 / (in_dim + out_dim))
            W = [[random.uniform(-limit, limit)
                  for _ in range(out_dim)]
                  for _ in range(in_dim)]
            b = [[0]*out_dim]

            self.layers.append([W, b])
            self.vel.append([
                [[0]*out_dim for _ in range(in_dim)],
                [[0]*out_dim]
            ])

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(self.layers):
            z = mat_add_bias(mat_mul(acts[-1], W), b)
            if i == len(self.layers)-1:
                a = softmax_matrix(z)
            else:
                a = tanh_matrix(z)
            acts.append(a)
        return acts

    def predict(self, x):
        return argmax_row(self.forward([x])[-1][0])

    def train(self, X, y, epochs=50, batch_size=8):
        n = len(X)
        for ep in range(epochs):
            for i in range(0, n, batch_size):
                Xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                acts = self.forward(Xb)
                delta = subtract_matrix(acts[-1], yb)
                bs = len(delta)

                for j in reversed(range(len(self.layers))):
                    a_prev = acts[j]
                    W, b = self.layers[j]
                    vW, vb = self.vel[j]

                    dW = mat_mul(transpose(a_prev), delta)
                    dW = [[v/bs for v in row] for row in dW]

                    db = [[sum(delta[r][c] for r in range(bs)) / bs
                           for c in range(len(delta[0]))]]

                    for r in range(len(W)):
                        for c in range(len(W[0])):
                            vW[r][c] = self.momentum*vW[r][c] - self.lr*dW[r][c]
                            W[r][c] += vW[r][c]

                    for c in range(len(b[0])):
                        vb[0][c] = self.momentum*vb[0][c] - self.lr*db[0][c]
                        b[0][c] += vb[0][c]

                    if j > 0:
                        delta = mat_mul(delta, transpose(W))
                        deriv = tanh_derivative_matrix(acts[j])
                        delta = [[delta[r][c]*deriv[r][c]
                                  for c in range(len(delta[0]))]
                                  for r in range(len(delta))]
            acts_full = self.forward(X)
            
            def cross_entropy(pred, target):
                eps = 1e-9
                loss = 0
                for i in range(len(pred)):
                    for j in range(len(pred[0])):
                        loss -= target[i][j]*math.log(pred[i][j]+eps)
                return loss/len(pred)
            loss = cross_entropy(acts_full[-1], y)
            print(f"Epoch {ep+1}/{epochs} completed | Loss: {loss:.4f}")

# =====================================================
# EVALUATION
# =====================================================

def evaluate(net, X, y_labels):
    correct = 0
    for i in range(len(X)):
        if net.predict(X[i]) == y_labels[i]:
            correct += 1
    return correct / len(X)

# =====================================================
# SIMPLE CNN (PURE PYTHON)
# =====================================================

def relu(x):
    return max(0, x)

def relu_deriv(x):
    return 1 if x > 0 else 0

def conv2d(image, kernel):
    h, w = 28, 28
    kh, kw = 3, 3
    output = [[0]*(w-2) for _ in range(h-2)]

    for i in range(h-2):
        for j in range(w-2):
            s = 0
            for ki in range(kh):
                for kj in range(kw):
                    s += image[i+ki][j+kj] * kernel[ki][kj]
            output[i][j] = relu(s)
    return output

def maxpool2x2(feature):
    h, w = len(feature), len(feature[0])
    pooled = [[0]*(w//2) for _ in range(h//2)]

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            pooled[i//2][j//2] = max(
                feature[i][j],
                feature[i+1][j],
                feature[i][j+1],
                feature[i+1][j+1]
            )
    return pooled

def flatten(feature):
    return [x for row in feature for x in row]

# =====================================================
# TRAIN WITH CNN FEATURE EXTRACTION
# =====================================================

kernel = [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(3)]

def extract_features(img_flat):
    img = [img_flat[i*28:(i+1)*28] for i in range(28)]
    conv = conv2d(img, kernel)
    pool = maxpool2x2(conv)
    return flatten(pool)

start = time.time()

X_train_raw = load_mnist_images("train-images.idx3-ubyte", limit=1000)
y_train_lbl = load_mnist_labels("train-labels.idx1-ubyte", limit=1000)
y_train = one_hot(y_train_lbl)

X_train = [extract_features(x) for x in X_train_raw]

X_test_raw = load_mnist_images("t10k-images.idx3-ubyte", limit=200)
y_test_lbl = load_mnist_labels("t10k-labels.idx1-ubyte", limit=200)

X_test = [extract_features(x) for x in X_test_raw]

net = FlexibleANN([169, 64, 32, 10], lr=0.01, momentum=0.9)
net.train(X_train, y_train, epochs=50, batch_size=8)

acc = evaluate(net, X_test, y_test_lbl)

print("\nTest Accuracy:", round(acc*100, 2), "%")
print("Elapsed time:", round(time.time()-start, 2), "seconds")
