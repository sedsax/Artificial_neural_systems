import tkinter as tk
from tkinter import ttk, messagebox
import math, random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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


def softmax(row):
    m = max(row)
    expv = [math.exp(x - m) for x in row]
    s = sum(expv)
    return [x / s for x in expv]


def transpose(A):
    return list(map(list, zip(*A)))


def subtract_matrix(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def scalar_mul_matrix(A, s):
    return [[x * s for x in row] for row in A]


def argmax_row(row):
    return max(range(len(row)), key=lambda i: row[i])


class SingleLayerANN:
    def __init__(self, input_dim, class_count, lr=0.01):
        self.lr = lr
        self.W = [[random.uniform(-0.5,0.5) for _ in range(class_count)] for _ in range(input_dim)]
        self.b = [[0 for _ in range(class_count)]]

    def forward(self, X):
        Z = mat_add_bias(mat_mul(X, self.W), self.b)
        return [softmax(row) for row in Z]

    def predict(self, Xsingle):
        probs = self.forward([Xsingle])[0]
        return argmax_row(probs)

    def train(self, X, Y, epochs):
        losses = []
        for _ in range(epochs):
            probs = self.forward(X)
            loss = 0

            # gradient: (p - y)
            grad = [[probs[i][j] - Y[i][j] for j in range(len(Y[0]))] for i in range(len(Y))]

            # MSE loss for visualization
            for i in range(len(Y)):
                for j in range(len(Y[0])):
                    loss += (probs[i][j] - Y[i][j])**2
            losses.append(loss / len(X))

            # backprop (single layer)
            dW = mat_mul(transpose(X), grad)
            db = [[sum(row[j] for row in grad) for j in range(len(self.b[0]))]]

            self.W = subtract_matrix(self.W, scalar_mul_matrix(dW, self.lr))
            self.b = subtract_matrix(self.b, scalar_mul_matrix(db, self.lr))

        return losses


class ANNVisualizer:
    def __init__(self, root):
        self.root = root
        root.title("Single Layer ANN — Classification Only")

        self.points = []
        self.test_mode = False
        self.net = None

        self.class_count = tk.IntVar(value=2)
        self.selected_class = tk.IntVar(value=0)
        self.lr_var = tk.DoubleVar(value=0.01)
        self.epoch_var = tk.IntVar(value=200)

        self.build_ui()
        self.draw_all()

    def build_ui(self):
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(left, text="Number of Classes").pack()
        ttk.Combobox(left, textvariable=self.class_count, values=[2,3,4,5], width=6).pack()

        tk.Label(left, text="Selected Class").pack()
        ttk.Combobox(left, textvariable=self.selected_class, values=[0,1,2,3,4], width=6).pack()

        tk.Label(left, text="Learning Rate").pack()
        tk.Entry(left, textvariable=self.lr_var, width=8).pack()

        tk.Label(left, text="Epochs").pack()
        tk.Entry(left, textvariable=self.epoch_var, width=8).pack()

        tk.Button(left, text="Train", command=self.train_model).pack(pady=4)
        tk.Button(left, text="Test Point", command=self.enable_test).pack(pady=4)
        tk.Button(left, text="Clear", command=self.clear_all).pack(pady=4)

        self.fig = plt.Figure(figsize=(9,5))
        self.ax_main = self.fig.add_subplot(2,1,1)
        self.ax_loss = self.fig.add_subplot(2,1,2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def enable_test(self):
        if not self.net:
            return messagebox.showwarning("Train first", "No model trained!")
        self.test_mode = True
        messagebox.showinfo("Test Mode", "Click to predict.")

    def clear_all(self):
        self.points.clear()
        self.net = None
        self.draw_all()

    def on_click(self, event):
        if event.inaxes != self.ax_main:
            return
        x,y = event.xdata, event.ydata

        if self.test_mode:
            pred = self.net.predict([x,y])
            self.points.append((x,y,pred,True))
            self.test_mode = False
        else:
            self.points.append((x,y,self.selected_class.get(),False))
        self.draw_all()

    def train_model(self):
        pts = [(x,y,c) for (x,y,c,t) in self.points if not t]
        if not pts:
            return messagebox.showerror("No data", "Add points first.")

        X = [[p[0],p[1]] for p in pts]
        Y = []

        C = self.class_count.get()
        for _,_,cls in pts:
            row = [0]*C
            row[cls] = 1
            Y.append(row)

        self.net = SingleLayerANN(2, C, self.lr_var.get())
        losses = self.net.train(X, Y, self.epoch_var.get())
        self.plot_losses(losses)
        self.draw_all()

    def plot_losses(self, losses):
        self.ax_loss.clear()
        self.ax_loss.plot(losses)
        self.ax_loss.set_title("Training Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("MSE")

    def draw_all(self):
        self.ax_main.clear()

        if self.net:
            self.draw_decision_boundary()

        for x,y,t,is_test in self.points:
            c = plt.cm.tab10(t%10)
            self.ax_main.scatter(x,y,c=[c],s=120 if is_test else 40,edgecolor='white')

        self.ax_main.set_xlim(-5,5)
        self.ax_main.set_ylim(-5,5)
        self.ax_main.grid(True)
        self.ax_main.set_title("Single Layer Classification")
        self.canvas.draw()

    def draw_decision_boundary(self):
        xs = [i/20 for i in range(-100,101)]
        ys = [i/20 for i in range(-100,101)]
        Z=[]
        for yy in ys:
            row=[]
            for xx in xs:
                row.append(self.net.predict([xx,yy]))
            Z.append(row)
        self.ax_main.contourf(xs,ys,Z,alpha=0.3,cmap=plt.cm.Pastel1)


root = tk.Tk()
ANNVisualizer(root)
root.mainloop()
