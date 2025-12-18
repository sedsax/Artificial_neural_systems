import numpy as np
import matplotlib.pyplot as plt

class SingleLayerFCN:
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))
        self.lr = lr

    def forward(self, X):
        return X @ self.W + self.b

    def train(self, X, y, epochs=200):
        errors = []
        for _ in range(epochs):
            y_pred = self.forward(X)
            error = np.mean((y_pred - y)**2)
            errors.append(error)

            dW = (2/X.shape[0]) * X.T @ (y_pred - y)
            db = (2/X.shape[0]) * np.sum(y_pred - y, axis=0, keepdims=True)

            self.W -= self.lr * dW
            self.b -= self.lr * db
        return errors

def plot_2d_data(X, y, title):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title(title)
    plt.show()
