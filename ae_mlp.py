
import math
import random
import struct
import time
import tkinter as tk

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
# FLEXIBLE ANN 
# =====================================================

class FlexibleANN:
    def save_weights(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.layers, f)

    def load_weights(self, filename):
        import pickle
        with open(filename, 'rb') as f:
                self.layers = pickle.load(f)
    def __init__(self, layer_sizes, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.layers = []
        self.vel = []

        for i in range(len(layer_sizes)-1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]

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
# AUTOENCODER (AE) + MLP ENTEGRASYONU
# =====================================================

import os

class Autoencoder:
    def __init__(self, input_dim=784, latent_dim=32, hidden_dim=128, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        # Encoder: input -> hidden -> latent
        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [[0]*hidden_dim]
        self.W2 = [[random.uniform(-0.1, 0.1) for _ in range(latent_dim)] for _ in range(hidden_dim)]
        self.b2 = [[0]*latent_dim]
        # Decoder: latent -> hidden -> output
        self.W3 = [[random.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in range(latent_dim)]
        self.b3 = [[0]*hidden_dim]
        self.W4 = [[random.uniform(-0.1, 0.1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b4 = [[0]*input_dim]
        # Momentum
        self.vW1 = [[0]*hidden_dim for _ in range(input_dim)]
        self.vb1 = [[0]*hidden_dim]
        self.vW2 = [[0]*latent_dim for _ in range(hidden_dim)]
        self.vb2 = [[0]*latent_dim]
        self.vW3 = [[0]*hidden_dim for _ in range(latent_dim)]
        self.vb3 = [[0]*hidden_dim]
        self.vW4 = [[0]*input_dim for _ in range(hidden_dim)]
        self.vb4 = [[0]*input_dim]

    def encoder(self, X):
        h1 = tanh_matrix(mat_add_bias(mat_mul(X, self.W1), self.b1))
        h2 = tanh_matrix(mat_add_bias(mat_mul(h1, self.W2), self.b2))
        return h2, h1

    def decoder(self, Z):
        h3 = tanh_matrix(mat_add_bias(mat_mul(Z, self.W3), self.b3))
        out = tanh_matrix(mat_add_bias(mat_mul(h3, self.W4), self.b4))
        return out, h3

    def forward(self, X):
        Z, h1 = self.encoder(X)
        out, h3 = self.decoder(Z)
        return out, Z, h1, h3

    def train(self, X, epochs=30, batch_size=16):
        n = len(X)
        losses = []
        for ep in range(epochs):
            for i in range(0, n, batch_size):
                Xb = X[i:i+batch_size]
                # Forward
                out, Z, h1, h3 = self.forward(Xb)
                # Loss: MSE
                loss = 0
                for r in range(len(Xb)):
                    for c in range(len(Xb[0])):
                        loss += (out[r][c] - Xb[r][c])**2
                loss /= len(Xb)
                # Backward (sadece W4, W3, W2, W1 için, momentumlu SGD)
                # dLoss/dOut
                dOut = [[2*(out[r][c] - Xb[r][c]) for c in range(len(Xb[0]))] for r in range(len(Xb))]
                # dOut/dW4
                dW4 = mat_mul(transpose(h3), dOut)
                dW4 = [[v/len(Xb) for v in row] for row in dW4]
                db4 = [[sum(dOut[r][c] for r in range(len(Xb))) / len(Xb) for c in range(len(Xb[0]))]]
                # dOut/dh3
                dh3 = mat_mul(dOut, transpose(self.W4))
                dh3 = [[dh3[r][c]*(1-h3[r][c]**2) for c in range(len(h3[0]))] for r in range(len(h3))]
                # dW3
                dW3 = mat_mul(transpose(Z), dh3)
                dW3 = [[v/len(Xb) for v in row] for row in dW3]
                db3 = [[sum(dh3[r][c] for r in range(len(Xb))) / len(Xb) for c in range(len(h3[0]))]]
                # dZ
                dZ = mat_mul(dh3, transpose(self.W3))
                dZ = [[dZ[r][c]*(1-Z[r][c]**2) for c in range(len(Z[0]))] for r in range(len(Z))]
                # dW2
                dW2 = mat_mul(transpose(h1), dZ)
                dW2 = [[v/len(Xb) for v in row] for row in dW2]
                db2 = [[sum(dZ[r][c] for r in range(len(Xb))) / len(Xb) for c in range(len(Z[0]))]]
                # dh1
                dh1 = mat_mul(dZ, transpose(self.W2))
                dh1 = [[dh1[r][c]*(1-h1[r][c]**2) for c in range(len(h1[0]))] for r in range(len(h1))]
                # dW1
                dW1 = mat_mul(transpose(Xb), dh1)
                dW1 = [[v/len(Xb) for v in row] for row in dW1]
                db1 = [[sum(dh1[r][c] for r in range(len(Xb))) / len(Xb) for c in range(len(h1[0]))]]
                # Update
                for r in range(len(self.W4)):
                    for c in range(len(self.W4[0])):
                        self.vW4[r][c] = self.momentum*self.vW4[r][c] - self.lr*dW4[r][c]
                        self.W4[r][c] += self.vW4[r][c]
                for c in range(len(self.b4[0])):
                    self.vb4[0][c] = self.momentum*self.vb4[0][c] - self.lr*db4[0][c]
                    self.b4[0][c] += self.vb4[0][c]
                for r in range(len(self.W3)):
                    for c in range(len(self.W3[0])):
                        self.vW3[r][c] = self.momentum*self.vW3[r][c] - self.lr*dW3[r][c]
                        self.W3[r][c] += self.vW3[r][c]
                for c in range(len(self.b3[0])):
                    self.vb3[0][c] = self.momentum*self.vb3[0][c] - self.lr*db3[0][c]
                    self.b3[0][c] += self.vb3[0][c]
                for r in range(len(self.W2)):
                    for c in range(len(self.W2[0])):
                        self.vW2[r][c] = self.momentum*self.vW2[r][c] - self.lr*dW2[r][c]
                        self.W2[r][c] += self.vW2[r][c]
                for c in range(len(self.b2[0])):
                    self.vb2[0][c] = self.momentum*self.vb2[0][c] - self.lr*db2[0][c]
                    self.b2[0][c] += self.vb2[0][c]
                for r in range(len(self.W1)):
                    for c in range(len(self.W1[0])):
                        self.vW1[r][c] = self.momentum*self.vW1[r][c] - self.lr*dW1[r][c]
                        self.W1[r][c] += self.vW1[r][c]
                for c in range(len(self.b1[0])):
                    self.vb1[0][c] = self.momentum*self.vb1[0][c] - self.lr*db1[0][c]
                    self.b1[0][c] += self.vb1[0][c]
            losses.append(loss)
            print(f"AE Epoch {ep+1}/{epochs} | Loss: {loss:.6f}")
        return losses

    def encode(self, X):
        Z, _ = self.encoder(X)
        return Z

    def decode(self, Z):
        out, _ = self.decoder(Z)
        return out

# =====================================================
# TRAIN WITH AE + MLP
# =====================================================

def train_and_save():
    start = time.time()
    X_train = load_mnist_images("train-images.idx3-ubyte", limit=10000)
    y_train_lbl = load_mnist_labels("train-labels.idx1-ubyte", limit=10000)
    y_train = one_hot(y_train_lbl)
    X_test = load_mnist_images("t10k-images.idx3-ubyte", limit=1000)
    y_test_lbl = load_mnist_labels("t10k-labels.idx1-ubyte", limit=1000)

    # 1. Autoencoder eğitimi
    ae = Autoencoder(input_dim=784, latent_dim=64, hidden_dim=128, lr=0.005, momentum=0.9)
    print("Autoencoder eğitiliyor...")
    ae_losses = ae.train(X_train, epochs=100, batch_size=8)

    # 2. MLP'yi, encoder'dan çıkan latent vektörlerle eğit
    print("MLP eğitiliyor...")
    X_train_latent = ae.encode(X_train)
    X_test_latent = ae.encode(X_test)
    net = FlexibleANN([64, 64, 32, 10], lr=0.005, momentum=0.9)
    net.train(X_train_latent, y_train, epochs=100, batch_size=8)
    acc = evaluate(net, X_test_latent, y_test_lbl)
    print("\nTest Accuracy:", round(acc*100, 2), "%")
    print("Elapsed time:", round(time.time()-start, 2), "seconds")
    # Ağırlıkları kaydet
    import pickle
    with open("ae_mlp_weights.pkl", "wb") as f:
        pickle.dump({"mlp": net.layers, "ae": {
            "W1": ae.W1, "b1": ae.b1, "W2": ae.W2, "b2": ae.b2,
            "W3": ae.W3, "b3": ae.b3, "W4": ae.W4, "b4": ae.b4
        }}, f)
    return net, ae, ae_losses

# =====================================================
# EVALUATION
# =====================================================

# Basit bir arayüz: El ile çizilen rakamı AE encoder ile latent'e çevirip MLP ile tahmin
class MNISTDrawWindow:
    def __init__(self, net, ae):
        self.net = net
        self.ae = ae
        self.cell = 10
        self.grid = [[0]*28 for _ in range(28)]

        win = tk.Toplevel()
        win.title("Draw MNIST Digit (AE+MLP)")

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
        latent, _ = self.ae.encoder([inp])
        probs = self.net.forward(latent)[-1][0]
        pred = max(range(len(probs)), key=lambda i: probs[i])
        self.label.config(
            text=f"Prediction: {pred}  (conf={round(max(probs),2)})"
        )

def main():
    if os.path.exists("ae_mlp_weights.pkl"):
        print("Ağırlık dosyası bulundu, yükleniyor...")
        with open("ae_mlp_weights.pkl", "rb") as f:
            import pickle
            data = pickle.load(f)
            net = FlexibleANN([32, 64, 32, 10], lr=0.01, momentum=0.9)
            net.layers = data["mlp"]
            ae = Autoencoder()
            ae.W1 = data["ae"]["W1"]
            ae.b1 = data["ae"]["b1"]
            ae.W2 = data["ae"]["W2"]
            ae.b2 = data["ae"]["b2"]
            ae.W3 = data["ae"]["W3"]
            ae.b3 = data["ae"]["b3"]
            ae.W4 = data["ae"]["W4"]
            ae.b4 = data["ae"]["b4"]
    else:
        net, ae, _ = train_and_save()

    root = tk.Tk()
    root.title("ANN Visualizer + MNIST (AE+MLP)")
    tk.Button(root, text="Draw MNIST Digit (AE+MLP)", command=lambda: MNISTDrawWindow(net, ae)).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    main()

