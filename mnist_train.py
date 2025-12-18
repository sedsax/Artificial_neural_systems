import tkinter as tk
from tkinter import ttk, messagebox
import math
import random
import struct
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =====================================================
# MNIST LOADER 
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
# MATRIX 
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
# SOFTMAX + LOSS
# =====================================================

def softmax(row):
    exps = [math.exp(x-max(row)) for x in row]
    s = sum(exps)
    return [e/s for e in exps]

def softmax_matrix(A):
    return [softmax(row) for row in A]

def cross_entropy(pred, target):
    eps = 1e-9
    loss = 0
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            loss -= target[i][j]*math.log(pred[i][j]+eps)
    return loss/len(pred)

def argmax_row(row):
    return max(range(len(row)), key=lambda i: row[i])

# =====================================================
# FLEXIBLE ANN 
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
# MNIST TRAIN
# =====================================================

def train_mnist_model():
    import time
    start = time.time()

    X_train = load_mnist_images("train-images.idx3-ubyte", limit=2000)
    y_train_lbl = load_mnist_labels("train-labels.idx1-ubyte", limit=2000)
    y_train = one_hot(y_train_lbl)

    X_test = load_mnist_images("t10k-images.idx3-ubyte", limit=200)
    y_test_lbl = load_mnist_labels("t10k-labels.idx1-ubyte", limit=200)

    net = FlexibleANN([784, 64, 32, 10], lr=0.01, momentum=0.9)
    net.train(X_train, y_train, epochs=50, batch_size=8)
    acc = evaluate(net, X_test, y_test_lbl)

    print("\nTest Accuracy:", round(acc*100, 2), "%")
    print("Elapsed time:", round(time.time()-start, 2), "seconds")

    return net

# =====================================================
# MNIST DRAW WINDOW
# =====================================================

class MNISTDrawWindow:
    def __init__(self, net):
        self.net = net
        self.cell = 10
        self.grid = [[0]*28 for _ in range(28)]

        win = tk.Toplevel()
        win.title("Draw MNIST Digit")

        self.canvas = tk.Canvas(win, width=280, height=280, bg="black")
        self.canvas.pack()

        self.label = tk.Label(win, text="Draw a digit (0–9)", font=("Arial", 12))
        self.label.pack()

        tk.Button(win, text="Predict", command=self.predict).pack()
        tk.Button(win, text="Clear", command=self.clear).pack()

        self.canvas.bind("<B1-Motion>", self.draw)

    def draw(self, e):
        x, y = e.x // 10, e.y // 10

        if 0 <= x < 28 and 0 <= y < 28:
            self.grid[y][x] = 1.0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < 28 and 0 <= nx < 28:
                    self.grid[ny][nx] = min(1.0, self.grid[ny][nx] + 0.1)

        self.canvas.create_rectangle(
            x*10, y*10, (x+1)*10, (y+1)*10,
            fill="white", outline="white"
        )


    def clear(self):
        self.canvas.delete("all")
        self.grid = [[0]*28 for _ in range(28)]

    def predict(self):
        inp = [v for row in self.grid for v in row]

        m = max(inp)
        if m > 0:
            inp = [v/m for v in inp]

        probs = self.net.forward([inp])[-1][0]
        pred = argmax_row(probs)
        self.label.config(
            text=f"Prediction: {pred}  (conf={round(max(probs),2)})"
        )

# =====================================================
# MAIN GUI
# =====================================================

class ANNVisualizer:
    def __init__(self, root):
        self.root = root
        root.title("ANN Visualizer + MNIST")

        self.mnist_net = train_mnist_model()

        tk.Button(root, text="Draw MNIST Digit",
                  command=lambda: MNISTDrawWindow(self.mnist_net)).pack(pady=20)

# =====================================================

if __name__ == "__main__":
    root = tk.Tk()
    ANNVisualizer(root)
    root.mainloop()
