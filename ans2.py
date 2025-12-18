import tkinter as tk
from tkinter import ttk, messagebox
import math
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =====================================================
# ===============  PURE PYTHON ANN  ===================
# =====================================================
# --- Matrix helpers (NO NUMPY) ---
def mat_mul(A, B):  # A: m×n, B: n×p
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            s = 0
            for k in range(len(B)):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def mat_add_bias(A, b):  # A: m×n, b: 1×n
    out = []
    for i in range(len(A)):
        row = [A[i][j] + b[0][j] for j in range(len(b[0]))]
        out.append(row)
    return out


def tanh_matrix(A):
    return [[math.tanh(x) for x in row] for row in A]


def tanh_derivative_matrix(A):  # A = tanh(z)
    return [[1 - x * x for x in row] for row in A]


def subtract_matrix(A, B):  # A-B
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def scalar_mul_matrix(A, s):
    return [[x * s for x in row] for row in A]


def transpose(A):
    return list(map(list, zip(*A)))


def mse_loss(output, target):
    s = 0
    count = len(output) * len(output[0])
    for i in range(len(output)):
        for j in range(len(output[0])):
            diff = output[i][j] - target[i][j]
            s += diff * diff
    return s / count


def argmax_row(row):
    max_i = 0
    for i in range(1, len(row)):
        if row[i] > row[max_i]:
            max_i = i
    return max_i


# =====================================================
# ========= PURE PYTHON TWO-LAYER ANN =================
# =====================================================
class SimpleANN:
    def __init__(self, input_dim, hidden, output_dim, lr=0.01):
        self.lr = lr

        # Random weights:
        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(hidden)] for _ in range(input_dim)]
        self.b1 = [[0 for _ in range(hidden)]]

        self.W2 = [[random.uniform(-0.5, 0.5) for _ in range(output_dim)] for _ in range(hidden)]
        self.b2 = [[0 for _ in range(output_dim)]]

    # ----------------------------------------------
    def forward(self, X):
        # X: m×2
        z1 = mat_add_bias(mat_mul(X, self.W1), self.b1)
        a1 = tanh_matrix(z1)
        z2 = mat_add_bias(mat_mul(a1, self.W2), self.b2)
        return a1, z2  # z2 = output

    # ----------------------------------------------
    def predict(self, Xsingle):
        # Xsingle: [x, y]
        a1, out = self.forward([Xsingle])
        return argmax_row(out[0])

    # ----------------------------------------------
    def train(self, X, y_onehot, epochs=50):
        losses = []

        for _ in range(epochs):
            a1, out = self.forward(X)

            # MSE loss
            loss = mse_loss(out, y_onehot)
            losses.append(loss)

            # BACKPROP
            grad_out = subtract_matrix(out, y_onehot)

            # dW2 = a1.T @ grad_out
            dW2 = mat_mul(transpose(a1), grad_out)

            # db2 = sum rows
            db2 = [[sum(grad_out[i][j] for i in range(len(grad_out))) 
                   for j in range(len(grad_out[0]))]]

            # da1 = grad_out @ W2.T
            da1 = mat_mul(grad_out, transpose(self.W2))

            # dz1 = da1 * tanh'(a1)
            deriv = tanh_derivative_matrix(a1)
            dz1 = [[da1[i][j] * deriv[i][j] for j in range(len(da1[0]))]
                   for i in range(len(da1))]

            # dW1 = X.T @ dz1
            dW1 = mat_mul(transpose(X), dz1)

            # db1 = sum rows
            db1 = [[sum(dz1[i][j] for i in range(len(dz1)))
                    for j in range(len(dz1[0]))]]

            # Update
            lr = self.lr
            self.W2 = subtract_matrix(self.W2, scalar_mul_matrix(dW2, lr))
            self.b2 = subtract_matrix(self.b2, scalar_mul_matrix(db2, lr))
            self.W1 = subtract_matrix(self.W1, scalar_mul_matrix(dW1, lr))
            self.b1 = subtract_matrix(self.b1, scalar_mul_matrix(db1, lr))

        return losses


# =====================================================
# =====================  GUI  =========================
# =====================================================
class ANNVisualizer:
    def __init__(self, root):
        self.root = root
        root.title("ANN Visualizer — NO NUMPY version")

        self.points = []      # (x, y, class, is_test)
        self.net = None
        self.test_mode = False

        self.build_ui()
        self.draw_all()

    # UI ------------------------------------------------
    def build_ui(self):
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(left, text="Classes").pack()
        self.class_count = tk.IntVar(value=2)
        ttk.Combobox(left, textvariable=self.class_count, values=[2, 3, 4, 5], width=5).pack()

        tk.Label(left, text="Selected class:").pack(pady=(10, 0))
        self.selected_class = tk.IntVar(value=0)
        ttk.Combobox(left, textvariable=self.selected_class,
                     values=[0, 1, 2, 3, 4], width=5).pack()

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
        tk.Button(left, text="Test point", command=self.enable_test_mode).pack(pady=5)
        tk.Button(left, text="Clear", command=self.clear_all).pack(pady=5)

        # FIGURE -------------------------------------------------
        self.fig = plt.Figure(figsize=(9, 5), dpi=100)
        self.ax_main = self.fig.add_subplot(2, 1, 1)
        self.ax_loss = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_click)

    # CLICK ADD ----------------------------------------
    def on_click(self, event):
        if event.inaxes != self.ax_main:
            return

        x, y = event.xdata, event.ydata

        # TEST MODE
        if self.test_mode:
            if self.net is None:
                messagebox.showwarning("No model", "Train first!")
                self.test_mode = False
                return

            pred = self.net.predict([x, y])
            self.points.append((x, y, pred, True))
            self.test_mode = False
            self.draw_all()
            return

        # TRAIN DATA
        cls = self.selected_class.get()
        if cls >= self.class_count.get():
            messagebox.showwarning("Class error", "Invalid class index.")
            return

        self.points.append((x, y, cls, False))
        self.draw_all()

    def enable_test_mode(self):
        self.test_mode = True
        messagebox.showinfo("Test", "Click on plot for test point.")

    def clear_all(self):
        self.points.clear()
        self.net = None
        self.draw_all()

    # TRAIN --------------------------------------------
    def train_model(self):
        train_pts = [(x, y, c) for (x, y, c, is_test) in self.points if not is_test]

        if not train_pts:
            messagebox.showerror("No data", "Add training points first!")
            return

        X = [[p[0], p[1]] for p in train_pts]
        y = [p[2] for p in train_pts]

        C = self.class_count.get()

        # one-hot
        y_onehot = []
        for cls in y:
            row = [0] * C
            row[cls] = 1
            y_onehot.append(row)

        self.net = SimpleANN(2, self.hidden_neurons.get(), C, lr=self.lr_var.get())
        losses = self.net.train(X, y_onehot, epochs=self.epoch_var.get())
        self.plot_losses(losses)
        self.draw_all()

    # PLOT LOSS ----------------------------------------
    def plot_losses(self, losses):
        self.ax_loss.clear()
        self.ax_loss.plot(losses)
        self.ax_loss.set_title("MSE loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")

    # DRAW ---------------------------------------------
    def draw_all(self):
        self.ax_main.clear()
        self.draw_decision_boundary()

        colors = plt.cm.tab10.colors

        for x, y, cls, is_test in self.points:
            c = colors[cls % 10]
            if is_test:
                self.ax_main.scatter(x, y, c=[c], s=120, edgecolor='black')
            else:
                self.ax_main.scatter(x, y, c=[c])

        self.ax_main.set_xlim(-5, 5)
        self.ax_main.set_ylim(-5, 5)
        self.ax_main.grid(True)
        self.ax_main.set_title("Click to add points")

        self.canvas.draw()

    # DECISION BOUNDARY -------------------------------
    def draw_decision_boundary(self):
        if self.net is None:
            return

        xs = [i / 20 for i in range(-100, 101)]
        ys = [i / 20 for i in range(-100, 101)]

        Z = []
        for y in ys:
            row = []
            for x in xs:
                row.append(self.net.predict([x, y]))
            Z.append(row)

        self.ax_main.contourf(xs, ys, Z, alpha=0.25, cmap=plt.cm.Pastel1)


# =====================================================
# RUN
# =====================================================
root = tk.Tk()
app = ANNVisualizer(root)
root.mainloop()
