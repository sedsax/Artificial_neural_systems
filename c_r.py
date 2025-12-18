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
    def __init__(self, input_dim, hidden, output_dim, lr=0.01):
        self.lr = lr
        self.W1 = [[random.uniform(-0.5,0.5) for _ in range(hidden)] for _ in range(input_dim)]
        self.b1 = [[0 for _ in range(hidden)]]
        self.W2 = [[random.uniform(-0.5,0.5) for _ in range(output_dim)] for _ in range(hidden)]
        self.b2 = [[0 for _ in range(output_dim)]]

    def forward(self, X):
        z1 = mat_add_bias(mat_mul(X, self.W1), self.b1)
        a1 = tanh_matrix(z1)
        z2 = mat_add_bias(mat_mul(a1, self.W2), self.b2)
        return a1, z2

    def predict_class(self, Xsingle):
        _, out = self.forward([Xsingle])
        return argmax_row(out[0])

    def predict_reg(self, Xsingle):
        _, out = self.forward([Xsingle])
        return out[0][0]

    def train(self, X, Y, epochs):
        losses = []
        for _ in range(epochs):
            a1, out = self.forward(X)
            loss = mse_loss(out, Y)
            losses.append(loss)

            grad = subtract_matrix(out, Y)
            dW2 = mat_mul(transpose(a1), grad)
            db2 = [[sum(grad[i][j] for i in range(len(grad)))
                    for j in range(len(grad[0]))]]
            da1 = mat_mul(grad, transpose(self.W2))
            dz1 = [[da1[i][j] * (1-a1[i][j]*a1[i][j])
                    for j in range(len(a1[0]))] for i in range(len(a1))]
            dW1 = mat_mul(transpose(X), dz1)
            db1 = [[sum(dz1[i][j] for i in range(len(dz1)))
                    for j in range(len(dz1[0]))]]

            lr = self.lr
            self.W2 = subtract_matrix(self.W2, scalar_mul_matrix(dW2, lr))
            self.b2 = subtract_matrix(self.b2, scalar_mul_matrix(db2, lr))
            self.W1 = subtract_matrix(self.W1, scalar_mul_matrix(dW1, lr))
            self.b1 = subtract_matrix(self.b1, scalar_mul_matrix(db1, lr))
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
                     values=[2,3,4,5], width=5).pack()

        # Class selection
        tk.Label(left, text="Selected class:").pack()
        self.selected_class = tk.IntVar(value=0)
        ttk.Combobox(left, textvariable=self.selected_class,
                     values=[0,1,2,3,4], width=5).pack()

        # Regression target input
        tk.Label(left, text="Regression target:").pack()
        self.regression_val = tk.DoubleVar(value=0.0)
        tk.Entry(left, textvariable=self.regression_val, width=7).pack()

        # Hyperparams
        tk.Label(left, text="Hidden neurons:").pack(pady=(10,0))
        self.hidden_neurons = tk.IntVar(value=12)
        tk.Entry(left, textvariable=self.hidden_neurons, width=8).pack()

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
        self.fig = plt.Figure(figsize=(9,5), dpi=100)
        self.ax_main = self.fig.add_subplot(2,1,1)
        self.ax_loss = self.fig.add_subplot(2,1,2)
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
        pts = [(x,y,t) for (x,y,t,is_test) in self.points if not is_test]
        if not pts:
            return messagebox.showerror("No data", "Add points first!")

        X = [[p[0], p[1]] for p in pts]
        Y = []
        mode = self.mode.get()

        if mode == "Classification":
            C = self.class_count.get()
            for _,_,cls in pts:
                row = [0]*C
                row[cls]=1
                Y.append(row)
            out_dim = C
        else:
            for _,_,val in pts:
                Y.append([val])
            out_dim = 1

        self.net = SimpleANN(2, self.hidden_neurons.get(), out_dim, self.lr_var.get())
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

        # Generate regression curve points
        xs = [i / 20 for i in range(-100, 101)]  # Generate x values from -5 to 5
        ys = []
        for x in xs:
            pred = self.net.predict_reg([x, 0])  # Predict using the trained model
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
