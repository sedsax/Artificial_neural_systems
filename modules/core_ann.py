import math
import random

# =====================
# MATRIX OPS
# =====================
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

# =====================
# SOFTMAX + LOSS
# =====================
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

# =====================
# FLEXIBLE ANN
# =====================
class FlexibleANN:
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
    def forward(self, X, regression=False):
        acts = [X]
        for i, (W, b) in enumerate(self.layers):
            z = mat_add_bias(mat_mul(acts[-1], W), b)
            if i == len(self.layers)-1:
                if regression:
                    a = z
                else:
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
    def save_weights(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.layers, f)
    def load_weights(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            self.layers = pickle.load(f)
