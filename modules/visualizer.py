import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ANNVisualizer for 2D data (ann_project)
class ANNVisualizer:
    def __init__(self, root, FlexibleANN, mat_mul, transpose, tanh_derivative_matrix, cross_entropy, subtract_matrix, softmax_matrix, argmax_row):
        self.root = root
        root.title("Flexible ANN Visualizer — Single / Multi Layer")

        self.points = []
        self.net = None
        self.test_mode = False

        self.FlexibleANN = FlexibleANN
        self.mat_mul = mat_mul
        self.transpose = transpose
        self.tanh_derivative_matrix = tanh_derivative_matrix
        self.cross_entropy = cross_entropy
        self.subtract_matrix = subtract_matrix
        self.softmax_matrix = softmax_matrix
        self.argmax_row = argmax_row

        self.build_ui()
        self.draw()

    def build_ui(self):
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, padx=10)

        tk.Label(left, text="Problem type").pack(pady=(10,0))
        self.problem_type = tk.StringVar(value="Classification")
        self.problem_type_box = ttk.Combobox(left, textvariable=self.problem_type, values=["Classification", "Regression"], width=14, state="readonly")
        self.problem_type_box.pack()
        self.problem_type.trace_add('write', self.on_problem_type_change)

        tk.Label(left,text="Classes").pack()
        self.class_count = tk.IntVar(value=2)
        self.class_box = ttk.Combobox(left,textvariable=self.class_count,
                                      values=[2,3,4,5],width=5)
        self.class_box.pack()
        self.class_box.bind("<<ComboboxSelected>>", self.update_class_list)

        tk.Label(left,text="Selected class").pack(pady=(10,0))
        self.selected_class = tk.IntVar(value=0)
        self.class_select = ttk.Combobox(left, textvariable=self.selected_class, width=5)
        self.class_select.pack()
        self.update_class_list()

        tk.Label(left, text="Layer type").pack(pady=(10,0))
        self.layer_type = tk.StringVar(value="Multi")
        self.layer_type_box = ttk.Combobox(left, textvariable=self.layer_type, values=["Single", "Multi"], width=8, state="readonly")
        self.layer_type_box.pack()
        self.layer_type.trace_add('write', self.on_layer_type_change)

        self.hidden_neurons = tk.IntVar(value=8)
        self.hidden_layers = tk.IntVar(value=1)
        self.lr = tk.DoubleVar(value=0.01)
        self.use_momentum = tk.BooleanVar(value=True)
        self.momentum = tk.DoubleVar(value=0.9)
        self.epochs = tk.IntVar(value=500)

        self.hidden_neurons_label = tk.Label(left, text="Hidden neurons")
        self.hidden_neurons_label.pack()
        self.hidden_neurons_entry = tk.Entry(left, textvariable=self.hidden_neurons, width=8)
        self.hidden_neurons_entry.pack()
        self.hidden_layers_label = tk.Label(left, text="Hidden layers")
        self.hidden_layers_label.pack()
        self.hidden_layers_entry = tk.Entry(left, textvariable=self.hidden_layers, width=8)
        self.hidden_layers_entry.pack()

        tk.Label(left, text="Learning rate").pack()
        tk.Entry(left, textvariable=self.lr, width=8).pack()
        tk.Checkbutton(left, text="Use momentum", variable=self.use_momentum).pack()
        tk.Label(left, text="Momentum").pack()
        tk.Entry(left, textvariable=self.momentum, width=8).pack()
        tk.Label(left, text="Epochs").pack()
        tk.Entry(left, textvariable=self.epochs, width=8).pack()

        tk.Button(left,text="Train",command=self.train).pack(pady=4)
        tk.Button(left,text="Test point",command=self.test_point).pack(pady=4)
        tk.Button(left,text="Clear",command=self.clear).pack(pady=4)

        self.fig = plt.Figure(figsize=(9,5))
        self.ax = self.fig.add_subplot(211)
        self.ax_loss = self.fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.fig,master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_problem_type_change(self, *args):
        if self.problem_type.get() == "Regression":
            self.class_box.config(state='disabled')
            self.class_select.config(state='disabled')
            self.class_count.set(1)
            self.selected_class.set(0)
        else:
            self.class_box.config(state='normal')
            self.class_select.config(state='normal')
            self.class_count.set(2)
            self.selected_class.set(0)

    def on_layer_type_change(self, *args):
        if self.layer_type.get() == "Single":
            self.hidden_layers.set(0)
            self.hidden_neurons.set(0)
            self.hidden_layers_entry.config(state='disabled')
            self.hidden_layers_label.config(state='disabled')
            self.hidden_neurons_entry.config(state='disabled')
            self.hidden_neurons_label.config(state='disabled')
        else:
            self.hidden_layers.set(1)
            self.hidden_neurons.set(8)
            self.hidden_layers_entry.config(state='normal')
            self.hidden_layers_label.config(state='normal')
            self.hidden_neurons_entry.config(state='normal')
            self.hidden_neurons_label.config(state='normal')

    def update_class_list(self, *_):
        C = self.class_count.get()
        self.class_select["values"] = list(range(C))
        self.selected_class.set(0)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if self.problem_type.get() == "Regression":
            self.points.append((x, y, 0, False))
        else:
            if self.test_mode:
                pred = self.net.predict([x, y])
                self.points.append((x, y, pred, True))
                self.test_mode = False
            else:
                self.points.append((x, y, self.selected_class.get(), False))
        self.draw()

    def train(self):
        X, y = [], []
        if self.problem_type.get() == "Regression":
            for x, yv, _, t in self.points:
                if not t:
                    X.append([x])
                    y.append([yv])
            out_dim = 1
        else:
            C = self.class_count.get()
            for x, yv, c, t in self.points:
                if not t:
                    if c >= C:
                        messagebox.showerror("Class error", "Invalid class label!")
                        return
                    X.append([x, yv])
                    one = [0] * C
                    one[c] = 1
                    y.append(one)
            out_dim = C

        if self.layer_type.get() == "Single":
            if self.problem_type.get() == "Regression":
                layers = [1, 1]
            else:
                layers = [2, out_dim]
        else:
            if self.problem_type.get() == "Regression":
                layers = [1] + [self.hidden_neurons.get()] * self.hidden_layers.get() + [1]
            else:
                layers = [2] + [self.hidden_neurons.get()] * self.hidden_layers.get() + [out_dim]

        momentum_val = self.momentum.get() if self.use_momentum.get() else 0.0
        self.net = self.FlexibleANN(layers, self.lr.get(), momentum_val)

        if self.problem_type.get() == "Regression":
            losses = self.train_regression(X, y, self.epochs.get())
        else:
            losses = self.net.train(X, y, self.epochs.get())

        self.ax_loss.clear()
        self.ax_loss.plot(losses)
        self.draw()

    def train_regression(self, X, y, epochs):
        losses = []
        for _ in range(epochs):
            acts = self.net.forward(X, regression=True)
            out = acts[-1]
            loss = sum((out[i][0] - y[i][0]) ** 2 for i in range(len(y))) / len(y)
            losses.append(loss)
            delta = [[out[i][0] - y[i][0]] for i in range(len(y))]
            for i in reversed(range(len(self.net.layers))):
                a_prev = acts[i]
                W, b = self.net.layers[i]
                vW, vb = self.net.vel[i]
                dW = self.mat_mul(self.transpose(a_prev), delta)
                db = [[sum(delta[r][c] for r in range(len(delta))) for c in range(len(delta[0]))]]
                for r in range(len(W)):
                    for c in range(len(W[0])):
                        vW[r][c] = self.net.momentum * vW[r][c] - self.net.lr * dW[r][c]
                        W[r][c] += vW[r][c]
                for c in range(len(b[0])):
                    vb[0][c] = self.net.momentum * vb[0][c] - self.net.lr * db[0][c]
                    b[0][c] += vb[0][c]
                if i > 0:
                    delta = self.mat_mul(delta, self.transpose(W))
                    deriv = self.tanh_derivative_matrix(acts[i])
                    delta = [[delta[r][c] * deriv[r][c] for c in range(len(delta[0]))] for r in range(len(delta))]
        return losses

    def draw(self):
        self.ax.clear()
        if self.problem_type.get() == "Regression":
            self.draw_regression_curve()
            for x, y, _, t in self.points:
                if t:
                    self.ax.scatter(x, y, c="red", s=120, edgecolor="black")
                else:
                    self.ax.scatter(x, y, c="blue")
        else:
            self.draw_boundary()
            colors = plt.cm.tab10.colors
            for x, y, c, t in self.points:
                if t:
                    self.ax.scatter(x, y, c=[colors[c]], s=120, edgecolor="black")
                else:
                    self.ax.scatter(x, y, c=[colors[c]])
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.grid(True)
        self.ax.set_title("Click to add points")
        self.canvas.draw()

    def draw_regression_curve(self):
        if not self.net:
            return
        xs = [i / 20 for i in range(-100, 101)]
        ys = []
        for x in xs:
            y_pred = self.net.forward([[x]], regression=True)[-1][0][0]
            ys.append(y_pred)
        self.ax.plot(xs, ys, color="orange", linewidth=2)

    def draw_boundary(self):
        if not self.net:
            return
        xs = [i/20 for i in range(-100,101)]
        ys = [i/20 for i in range(-100,101)]
        Z = []
        for y in ys:
            row = []
            for x in xs:
                probs = self.net.forward([[x,y]])[-1][0]  # softmax
                pred_class = max(range(len(probs)), key=lambda i: probs[i])
                row.append(pred_class)
            Z.append(row)
        self.ax.contourf(xs, ys, Z, levels=len(self.class_select["values"]), cmap=plt.cm.Pastel1, alpha=0.4)

    def test_point(self):
        self.test_mode=True

    def clear(self):
        self.points.clear()
        self.net=None
        self.draw()

# MNISTDrawWindow for MNIST digit drawing (mnist_train)
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
        pred = max(range(len(probs)), key=lambda i: probs[i])
        self.label.config(
            text=f"Prediction: {pred}  (conf={round(max(probs),2)})"
        )
