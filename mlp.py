import tkinter as tk
from tkinter import ttk, messagebox
import math
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =====================================================
# ===============  PURE PYTHON ANN  ===================
# =====================================================
def mat_mul(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            s = 0
            for k in range(len(B)):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result

def mat_add_bias(A, b):
    return [[A[i][j] + b[0][j] for j in range(len(b[0]))] for i in range(len(A))]

def tanh_matrix(A):
    return [[math.tanh(x) for x in row] for row in A]

def tanh_derivative_matrix(A):
    return [[1 - x*x for x in row] for row in A]

def subtract_matrix(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def scalar_mul_matrix(A, s):
    return [[x*s for x in row] for row in A]

def transpose(A):
    return list(map(list, zip(*A)))

def mse_loss(output, target):
    s = 0
    count = len(output)*len(output[0])
    for i in range(len(output)):
        for j in range(len(output[0])):
            diff = output[i][j] - target[i][j]
            s += diff * diff
    return s / count

def argmax_row(row):
    return max(range(len(row)), key=lambda i: row[i])

# =====================================================
# ========= ANN (Classification + Regression) =========
# =====================================================
class SimpleANN:
    def __init__(self, input_dim, hidden_layers, output_dim, lr=0.01, activation="tanh"):
        self.lr = lr
        self.hidden_layers = hidden_layers
        self.activation = activation

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        layer_dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(layer_dims) - 1):
            self.weights.append(
                [[random.uniform(-0.5, 0.5) for _ in range(layer_dims[i + 1])] for _ in range(layer_dims[i])]
            )
            self.biases.append([[0 for _ in range(layer_dims[i + 1])]])

    def activate(self, x):
        if self.activation == "tanh":
            return math.tanh(x)
        elif self.activation == "sigmoid":
            return 1 / (1 + math.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def activate_derivative(self, x):
        if self.activation == "tanh":
            return 1 - x * x
        elif self.activation == "sigmoid":
            return x * (1 - x)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        activations = [X]
        for W, b in zip(self.weights, self.biases):
            z = mat_add_bias(mat_mul(activations[-1], W), b)
            a = [[self.activate(val) for val in row] for row in z]
            activations.append(a)
        return activations

    def predict_class(self, Xsingle):
        activations = self.forward([Xsingle])
        return argmax_row(activations[-1][0])

    def predict_reg(self, Xsingle):
        activations = self.forward([Xsingle])
        return activations[-1][0][0]

    def train(self, X, Y, epochs):
        losses = []
        for _ in range(epochs):
            activations = self.forward(X)
            out = activations[-1]
            loss = mse_loss(out, Y)
            losses.append(loss)

            # Backpropagation
            grad = subtract_matrix(out, Y)
            for i in range(len(self.weights) - 1, -1, -1):
                dW = mat_mul(transpose(activations[i]), grad)

                # db as 1 x units row to match self.biases[i]
                db_row = [sum(grad[r][c] for r in range(len(grad))) for c in range(len(self.biases[i][0]))]
                db = [db_row]

                if i > 0:
                    grad = mat_mul(grad, transpose(self.weights[i]))
                    grad = [[grad[r][c] * self.activate_derivative(activations[i][r][c]) for c in range(len(grad[0]))] for r in range(len(grad))]

                self.weights[i] = subtract_matrix(self.weights[i], scalar_mul_matrix(dW, self.lr))
                self.biases[i] = subtract_matrix(self.biases[i], scalar_mul_matrix(db, self.lr))

        return losses

# =====================================================
# ====================== GUI ==========================
# =====================================================
class ANNVisualizer:
    def __init__(self, root):
        self.root = root
        root.title("ANN Visualizer — Classification & Regression")

        self.points = []  # (x, y, target, is_test)
        self.test_mode = False
        self.net = None
        self.mode = tk.StringVar(value="Classification")

        self.build_ui()
        self.draw_all()

    def build_ui(self):
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, padx=10, pady=10)

        # Mode Selector
        tk.Label(left, text="Mode:").pack()
        ttk.Combobox(left, textvariable=self.mode,
                     values=["Classification", "Regression"],
                     width=12).pack()

        # Class count
        tk.Label(left, text="Classes (only for classification)").pack()
        self.class_count = tk.IntVar(value=2)
        ttk.Combobox(left, textvariable=self.class_count,
                     values=[2, 3, 4, 5], width=5).pack()

        # Class selection
        tk.Label(left, text="Selected class:").pack()
        self.selected_class = tk.IntVar(value=0)
        ttk.Combobox(left, textvariable=self.selected_class,
                     values=[0, 1, 2, 3, 4], width=5).pack()

        # Regression target input
        tk.Label(left, text="Regression target:").pack()
        self.regression_val = tk.DoubleVar(value=0.0)
        tk.Entry(left, textvariable=self.regression_val, width=7).pack()

        # Hyperparams
        tk.Label(left, text="Hidden layers (comma-separated):").pack(pady=(10, 0))
        self.hidden_layers = tk.StringVar(value="12")
        tk.Entry(left, textvariable=self.hidden_layers, width=15).pack()

        tk.Label(left, text="Activation function:").pack()
        self.activation_func = tk.StringVar(value="tanh")
        ttk.Combobox(left, textvariable=self.activation_func,
                     values=["tanh", "sigmoid"], width=10).pack()

        tk.Label(left, text="Learning rate:").pack()
        self.lr_var = tk.DoubleVar(value=0.01)
        tk.Entry(left, textvariable=self.lr_var, width=8).pack()

        tk.Label(left, text="Epochs:").pack()
        self.epoch_var = tk.IntVar(value=100)
        tk.Entry(left, textvariable=self.epoch_var, width=8).pack()

        tk.Button(left, text="Train", command=self.train_model).pack(pady=5)
        tk.Button(left, text="Test point", command=self.enable_test).pack(pady=5)
        tk.Button(left, text="Clear", command=self.clear_all).pack(pady=5)

        # Plot setup
        self.fig = plt.Figure(figsize=(9, 5), dpi=100)
        self.ax_main = self.fig.add_subplot(2, 1, 1)
        self.ax_loss = self.fig.add_subplot(2, 1, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def enable_test(self):
        self.test_mode = True
        messagebox.showinfo("Test", "Click anywhere on the plot.")

    def clear_all(self):
        self.points.clear()
        self.net = None
        self.draw_all()

    def on_click(self, event):
        if event.inaxes != self.ax_main:
            return
        x, y = event.xdata, event.ydata

        if self.test_mode:
            if not self.net:
                messagebox.showwarning("Train first", "No model trained!")
                return
            self.test_prediction(x,y)
            self.test_mode = False
            return

        if self.mode.get() == "Classification":
            cls = self.selected_class.get()
            self.points.append((x,y,cls,False))
        else:
            r = self.regression_val.get()
            self.points.append((x,y,r,False))
        self.draw_all()

    def test_prediction(self, x, y):
        if self.mode.get() == "Classification":
            pred = self.net.predict_class([x,y])
            self.points.append((x,y,pred,True))
        else:
            pred = self.net.predict_reg([x,y])
            self.points.append((x,y,pred,True))
            messagebox.showinfo("Prediction", f"Predicted Y: {pred:.3f}")
        self.draw_all()

    def train_model(self):
        pts = [(x, y, t) for (x, y, t, is_test) in self.points if not is_test]
        if not pts:
            return messagebox.showerror("No data", "Add points first!")

        X = [[p[0], p[1]] for p in pts]
        Y = []
        mode = self.mode.get()

        if mode == "Classification":
            C = self.class_count.get()
            for _, _, cls in pts:
                row = [0] * C
                row[cls] = 1
                Y.append(row)
            out_dim = C
        else:
            for _, _, val in pts:
                Y.append([val])
            out_dim = 1

        hidden_layers = [int(x) for x in self.hidden_layers.get().split(",")]
        self.net = SimpleANN(2, hidden_layers, out_dim, self.lr_var.get(), self.activation_func.get())
        losses = self.net.train(X, Y, self.epoch_var.get())
        self.plot_losses(losses)
        self.draw_all()

    def plot_losses(self, losses):
        self.ax_loss.clear()
        self.ax_loss.plot(losses)
        self.ax_loss.set_title("MSE Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")

    def draw_all(self):
        self.ax_main.clear()

        if self.net:
            if self.mode.get() == "Classification":
                self.draw_class_boundary()
            else:
                self.draw_reg_surface()

        for x,y,t,is_test in self.points:
            if self.mode.get() == "Classification":
                c = plt.cm.tab10(t%10)
            else:
                c = 'black'
            self.ax_main.scatter(x,y,c=[c],s=120 if is_test else 40,edgecolor='white')

        self.ax_main.set_xlim(-5,5)
        self.ax_main.set_ylim(-5,5)
        self.ax_main.grid(True)
        self.ax_main.set_title(self.mode.get())
        self.canvas.draw()

    def draw_class_boundary(self):
        xs = [i/20 for i in range(-100,101)]
        ys = [i/20 for i in range(-100,101)]
        Z = []
        for yy in ys:
            row=[]
            for xx in xs:
                row.append(self.net.predict_class([xx,yy]))
            Z.append(row)
        self.ax_main.contourf(xs,ys,Z,alpha=0.3,cmap=plt.cm.Pastel1)

    def draw_reg_surface(self):
        # Clear the plot
        self.ax_main.clear()

        # Ensure the model is trained
        if not self.net:
            self.ax_main.set_xlim(-5, 5)
            self.ax_main.set_ylim(-5, 5)
            self.ax_main.grid(True)
            self.ax_main.set_title("Regression")
            self.canvas.draw()
            return

        # Generate regression curve points (vary x, keep second input at 0)
        xs = [i / 20 for i in range(-100, 101)]  # -5 .. 5
        ys = []
        for xv in xs:
            pred = self.net.predict_reg([xv, 0])
            ys.append(pred)

        # Plot regression curve
        self.ax_main.plot(xs, ys, color="red", label="Regression Curve")

        # Scatter points
        for x, y, t, is_test in self.points:
            self.ax_main.scatter(x, y, color="black" if not is_test else "blue", s=120 if is_test else 40, edgecolor="white")

        self.ax_main.set_xlim(-5, 5)
        self.ax_main.set_ylim(-5, 5)
        self.ax_main.grid(True)
        self.ax_main.set_title("Regression")
        self.ax_main.legend()
        self.canvas.draw()


root = tk.Tk()
app = ANNVisualizer(root)
root.mainloop()
