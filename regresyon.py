import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# =========================
# PURE PYTHON MATRIX OPS
# =========================

def transpose(M):
    return list(map(list, zip(*M)))

def matmul(A, B):
    result = [[0]*len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            s = 0
            for k in range(len(B)):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result

def invert_matrix(M):
    n = len(M)
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    M = [row[:] for row in M]

    for i in range(n):
        diag = M[i][i]
        if diag == 0:
            raise ValueError("Matrix not invertible")

        for j in range(n):
            M[i][j] /= diag
            I[i][j] /= diag

        for r in range(n):
            if r != i:
                factor = M[r][i]
                for c in range(n):
                    M[r][c] -= factor * M[i][c]
                    I[r][c] -= factor * I[i][c]

    return I

def vector_dot(a, b):
    return sum(x*y for x,y in zip(a,b))


# =========================
# REGRESSION MODEL
# =========================

class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.weights = None

    def design_matrix(self, X):
        Phi = []
        for x in X:
            Phi.append([x**d for d in range(self.degree + 1)])
        return Phi

    def train(self, X, Y):
        Phi = self.design_matrix(X)
        Y = [[y] for y in Y]

        Phi_T = transpose(Phi)
        A = matmul(Phi_T, Phi)
        A_inv = invert_matrix(A)
        B = matmul(Phi_T, Y)
        self.weights = matmul(A_inv, B)

    def predict(self, x):
        terms = [x**d for d in range(self.degree + 1)]
        return vector_dot(terms, [w[0] for w in self.weights])


# =========================
# GUI
# =========================

class RegApp:
    def __init__(self, root):
        self.root = root
        root.title("Regression Visualizer — Linear & Polynomial (Pure Python)")

        self.points = []
        self.model = None
        self.test_mode = False

        self.build_ui()
        self.draw()

    def build_ui(self):
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(left, text="Polynomial Degree").pack()
        self.degree = tk.IntVar(value=1)
        tk.Spinbox(left, from_=1, to=10, textvariable=self.degree, width=5).pack()

        tk.Button(left, text="Train", command=self.train).pack(pady=5)
        tk.Button(left, text="Test Point", command=self.enable_test).pack(pady=5)
        tk.Button(left, text="Clear", command=self.clear).pack(pady=5)

        self.fig = plt.Figure(figsize=(9, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def enable_test(self):
        self.test_mode = True
        messagebox.showinfo("Test", "Click on graph to predict Y")

    def clear(self):
        self.points.clear()
        self.model = None
        self.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if self.test_mode:
            if not self.model:
                return messagebox.showerror("Error", "Train first!")

            pred = self.model.predict(x)
            self.points.append((x, pred, True))
            messagebox.showinfo("Prediction", f"Predicted Y = {pred:.3f}")
            self.test_mode = False
        else:
            self.points.append((x, y, False))

        self.draw()

    def train(self):
        if len(self.points) < 2:
            return messagebox.showerror("Error", "Add more data points")

        X = [p[0] for p in self.points if not p[2]]
        Y = [p[1] for p in self.points if not p[2]]

        self.model = PolynomialRegression(self.degree.get())
        self.model.train(X, Y)

        self.draw()

    def draw(self):
        self.ax.clear()

        # Points
        for x, y, is_test in self.points:
            self.ax.scatter(x, y, color="red" if is_test else "black", s=120 if is_test else 50)

        # Regression curve
        if self.model:
            xs = [i/20 for i in range(-100, 101)]
            ys = [self.model.predict(x) for x in xs]
            self.ax.plot(xs, ys, color="blue", linewidth=2)

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.grid(True)

        title = "Linear Regression" if self.degree.get() == 1 else f"Polynomial Regression (Degree {self.degree.get()})"
        self.ax.set_title(title)

        self.canvas.draw()


root = tk.Tk()
app = RegApp(root)
root.mainloop()
