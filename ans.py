import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -------------------------------
# ANN MODEL (2 layer tanh network)
# -------------------------------
class SimpleANN:
    def __init__(self, input_dim, hidden, output_dim, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden) * 0.5
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, output_dim) * 0.5
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = z2  # linear output
        return a1, a2

    def predict(self, X):
        _, out = self.forward(X)
        return np.argmax(out, axis=1)

    def train(self, X, y_onehot, epochs=50):
        losses = []
        for _ in range(epochs):
            a1, out = self.forward(X)

            loss = np.mean((out - y_onehot) ** 2)
            losses.append(loss)

            grad_out = (out - y_onehot)

            dW2 = a1.T @ grad_out
            db2 = np.sum(grad_out, axis=0)

            da1 = grad_out @ self.W2.T
            dz1 = da1 * (1 - a1 ** 2)

            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0)

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

        return losses


# ----------------------------------------------
# GUI CLASS
# ----------------------------------------------
class ANNVisualizer:
    def __init__(self, root):
        self.root = root
        root.title("ANN Visualizer — Classification & MSE graph")

        self.points = []  # (x, y, class, is_test)
        self.net = None
        self.test_mode = False

        self.build_ui()
        self.draw_all()

    # -----------------------------
    # UI
    # -----------------------------
    def build_ui(self):
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(left, text="Classes").pack()
        self.class_count = tk.IntVar(value=2)
        ttk.Combobox(left, textvariable=self.class_count, values=[2, 3, 4, 5], width=5).pack()

        tk.Label(left, text="Selected class for adding:").pack(pady=(10, 0))
        self.selected_class = tk.IntVar(value=0)
        ttk.Combobox(left, textvariable=self.selected_class, values=list(range(5)), width=5).pack()

        tk.Label(left, text="Hidden neurons:").pack(pady=(10, 0))
        self.hidden_neurons = tk.IntVar(value=12)
        tk.Entry(left, textvariable=self.hidden_neurons, width=8).pack()

        tk.Label(left, text="Learning rate:").pack(pady=(10, 0))
        self.lr_var = tk.DoubleVar(value=0.01)
        tk.Entry(left, textvariable=self.lr_var, width=8).pack()

        tk.Label(left, text="Epochs:").pack(pady=(10, 0))
        self.epoch_var = tk.IntVar(value=100)
        tk.Entry(left, textvariable=self.epoch_var, width=8).pack()

        tk.Button(left, text="Train", command=self.train_model).pack(pady=5)
        tk.Button(left, text="Add test point (click plot)", command=self.enable_test_mode).pack(pady=5)
        tk.Button(left, text="Clear", command=self.clear_all).pack(pady=5)

        # --------------------------------------------
        # MATPLOTLIB FIGURES
        # --------------------------------------------
        self.fig = plt.Figure(figsize=(9, 5), dpi=100)
        self.ax_main = self.fig.add_subplot(2, 1, 1)
        self.ax_loss = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    # ---------------------------------------------------
    # Clicking to add training or test point
    # ---------------------------------------------------
    def on_click(self, event):
        if event.inaxes != self.ax_main:
            return

        x, y = event.xdata, event.ydata

        # -------- TEST MODE ----------
        if self.test_mode:
            if self.net is None:
                messagebox.showwarning("No model", "Train the model first.")
                self.test_mode = False
                return

            pred = self.net.predict(np.array([[x, y]]))[0]
            self.points.append((x, y, pred, True))

            self.test_mode = False
            self.draw_all()
            return

        # -------- TRAIN MODE ----------
        cls = self.selected_class.get()
        if cls >= self.class_count.get():
            messagebox.showwarning("Class error", "Selected class exceeds class count.")
            return

        self.points.append((x, y, cls, False))
        self.draw_all()

    # ---------------------------------------------------
    def enable_test_mode(self):
        self.test_mode = True
        messagebox.showinfo("Test Mode", "Click on the plot. The model will classify the point.")

    # ---------------------------------------------------
    def clear_all(self):
        self.points.clear()
        self.net = None
        self.draw_all()

    # ---------------------------------------------------
    # TRAINING
    # ---------------------------------------------------
    def train_model(self):
        train_pts = [(x, y, c) for (x, y, c, is_test) in self.points if not is_test]
        if not train_pts:
            messagebox.showerror("No data", "Please add training points.")
            return

        X = np.array([[p[0], p[1]] for p in train_pts])
        y = np.array([p[2] for p in train_pts])

        C = self.class_count.get()
        y_onehot = np.zeros((len(y), C))
        y_onehot[np.arange(len(y)), y] = 1

        self.net = SimpleANN(2, self.hidden_neurons.get(), C, lr=self.lr_var.get())
        losses = self.net.train(X, y_onehot, epochs=self.epoch_var.get())
        self.plot_losses(losses)
        self.draw_all()

    # ---------------------------------------------------
    def plot_losses(self, losses):
        self.ax_loss.clear()
        self.ax_loss.plot(losses)
        self.ax_loss.set_title("MSE loss vs epochs")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")

    # ---------------------------------------------------
    def draw_all(self):
        self.ax_main.clear()
        self.draw_decision_boundary()

        # Draw points
        C = self.class_count.get()
        colors = plt.cm.tab10.colors

        for x, y, cls, is_test in self.points:
            c = colors[cls % 10]
            if is_test:
                self.ax_main.scatter(x, y, edgecolor='black', s=120, c=[c])  
            else:
                self.ax_main.scatter(x, y, c=[c])

        self.ax_main.set_title("2D coordinate plane — click to add points")
        self.ax_main.set_xlim(-5, 5)
        self.ax_main.set_ylim(-5, 5)
        self.ax_main.grid(True)

        self.canvas.draw()

    # ---------------------------------------------------
    def draw_decision_boundary(self):
        if self.net is None:
            return

        xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = self.net.predict(grid)
        preds = preds.reshape(xx.shape)

        self.ax_main.contourf(xx, yy, preds, alpha=0.25, cmap=plt.cm.Pastel1)


# ------------------------
# RUN
# ------------------------
root = tk.Tk()
app = ANNVisualizer(root)
root.mainloop()
