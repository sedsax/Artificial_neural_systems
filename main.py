import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.widgets import Button, RadioButtons, TextBox, Slider  # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore
import tkinter as tk
from tkinter import ttk
import time

# --- Aktivasyon fonksiyonları ---
def activation_fn(x, activation="sigmoid"):
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif activation == "relu":
        return np.maximum(0, x)
    elif activation == "tanh":
        return np.tanh(x)
    elif activation == "linear":
        return x
    else:
        return x

def activation_derivative(y, activation="sigmoid"):
    if activation == "sigmoid":
        return y * (1 - y)
    elif activation == "relu":
        return np.where(y > 0, 1, 0)
    elif activation == "tanh":
        return 1 - y ** 2
    elif activation == "linear":
        return 1
    else:
        return 1

# --- Eğitim fonksiyonu ---
def train_live(x_samples, y_true, Size, Dim, x_weights, x_bias,
               activation="sigmoid", learning_const=0.1,
               Max_error=0.01, Max_epoch=1000, on_update=None):

    weights = np.array(x_weights).reshape(Dim, 1)
    bias = np.array(x_bias).reshape(1)
    
    for epoch in range(Max_epoch):
        net = np.dot(x_samples, weights) + bias
        output = activation_fn(net, activation)
        error = y_true - output
        mse = np.mean(error ** 2)
        delta = error * activation_derivative(output, activation)
        weights += learning_const * np.dot(x_samples.T, delta) / Size
        bias += learning_const * np.sum(delta) / Size

        if on_update and epoch % 20 == 0:
            on_update(weights, bias, epoch, mse)
            root.update_idletasks()
            time.sleep(0.05)

        if mse < Max_error:
            break

    if on_update:
        on_update(weights, bias, epoch, mse)
    return weights, bias

# --- GUI sınıfı ---
class PerceptronGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptron Görselleştirici (Classification & Regression)")

        # Veri
        self.class0 = []
        self.class1 = []
        self.current_class = tk.StringVar(value="Class 0")
        self.mode = tk.StringVar(value="Classification")

        # Ana çerçeve
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        # Sol: Grafik
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side="left", fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.axhline(0, color='gray', linewidth=1)
        self.ax.axvline(0, color='gray', linewidth=1)
        self.ax.grid(True)
        self.ax.set_title("Koordinat düzlemi (nokta eklemek için tıklayın)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self.onclick)

        # Sağ: Kontroller
        control_frame = tk.Frame(main_frame, padx=10, pady=10)
        control_frame.pack(side="right", fill="y")

        # Mod seçimi
        tk.Label(control_frame, text="Mod Seçimi:", font=("Arial", 11, "bold")).pack(anchor="w", pady=5)
        mode_box = ttk.Combobox(control_frame, textvariable=self.mode, values=["Classification","Regression"], state="readonly")
        mode_box.pack(fill="x")
        mode_box.bind("<<ComboboxSelected>>", self.mode_changed)

        # Class seçimi (yalnızca Classification)
        tk.Label(control_frame, text="Sınıf Seçimi:", font=("Arial", 11, "bold")).pack(anchor="w", pady=5)
        self.class_box = ttk.Combobox(control_frame, textvariable=self.current_class, values=["Class 0", "Class 1"], state="readonly")
        self.class_box.pack(fill="x")

        # Aktivasyon tipi
        tk.Label(control_frame, text="Aktivasyon:", font=("Arial", 11, "bold")).pack(anchor="w", pady=5)
        self.activation_choice = ttk.Combobox(control_frame, values=["sigmoid", "tanh", "relu", "linear"], state="readonly")
        self.activation_choice.current(0)
        self.activation_choice.pack(fill="x")

        # Öğrenme oranı ve epoch
        tk.Label(control_frame, text="Öğrenme Katsayısı (η):", font=("Arial", 11, "bold")).pack(anchor="w", pady=5)
        self.lr_entry = tk.Entry(control_frame)
        self.lr_entry.insert(0, "0.1")
        self.lr_entry.pack(fill="x")

        tk.Label(control_frame, text="Epoch Sayısı:", font=("Arial", 11, "bold")).pack(anchor="w", pady=5)
        self.epoch_entry = tk.Entry(control_frame)
        self.epoch_entry.insert(0, "1000")
        self.epoch_entry.pack(fill="x")

        tk.Button(control_frame, text="Eğit (Canlı)", command=self.train_and_plot, bg="#4CAF50", fg="white").pack(pady=10, fill="x")
        tk.Button(control_frame, text="Temizle", command=self.reset, bg="#E53935", fg="white").pack(pady=5, fill="x")

        self.status_label = tk.Label(control_frame, text="", fg="blue", font=("Arial", 10))
        self.status_label.pack(pady=10)

        self.mode_changed()

    def mode_changed(self, event=None):
        if self.mode.get() == "Regression":
            self.class_box.config(state="disabled")
            self.activation_choice.set("linear")
        else:
            self.class_box.config(state="readonly")
            self.activation_choice.set("sigmoid")

    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        if self.mode.get() == "Classification":
            if self.current_class.get() == "Class 0":
                self.class0.append([x, y])
                self.ax.scatter(x, y, color='red')
            else:
                self.class1.append([x, y])
                self.ax.scatter(x, y, color='blue')
        else:
            self.class0.append([x, y])
            self.ax.scatter(x, y, color='green')
        self.canvas.draw()

    def reset(self):
        self.class0 = []
        self.class1 = []
        self.ax.clear()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.axhline(0, color='gray', linewidth=1)
        self.ax.axvline(0, color='gray', linewidth=1)
        self.ax.grid(True)
        self.ax.set_title("Koordinat düzlemi (nokta eklemek için tıklayın)")
        self.status_label.config(text="")
        self.canvas.draw()

    def train_and_plot(self):
        if self.mode.get() == "Classification":
            if len(self.class0) == 0 or len(self.class1) == 0:
                self.status_label.config(text="⚠️ Her iki sınıftan da nokta ekleyin!")
                return
            X = np.vstack((self.class0, self.class1))
            y = np.array([[0]] * len(self.class0) + [[1]] * len(self.class1))
        else:  # Regression
            if len(self.class0) == 0:
                self.status_label.config(text="⚠️ Nokta ekleyin!")
                return
            X = np.array(self.class0)[:,0].reshape(-1,1)
            y = np.array(self.class0)[:,1].reshape(-1,1)

        weights = np.random.randn(X.shape[1], 1) * 0.1
        bias = np.zeros((1,))
        activation = self.activation_choice.get()
        learning_rate = float(self.lr_entry.get())
        max_epoch = int(self.epoch_entry.get())

        self.ax.clear()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.axhline(0, color='gray', linewidth=1)
        self.ax.axvline(0, color='gray', linewidth=1)
        self.ax.grid(True)

        # Noktaları çiz
        if self.mode.get() == "Classification":
            self.ax.scatter(np.array(self.class0)[:,0], np.array(self.class0)[:,1], color='red', label='Class 0')
            self.ax.scatter(np.array(self.class1)[:,0], np.array(self.class1)[:,1], color='blue', label='Class 1')
        else:
            self.ax.scatter(X[:,0], y[:,0], color='green', label='Veri')
        self.ax.legend()
        self.canvas.draw()

        def update_line(w, b, epoch, mse):
            # Sadece önceki çizgileri temizle
            for line in self.ax.lines[:]:
                line.remove()
            # Noktaları tekrar çiz
            if self.mode.get() == "Classification":
                self.ax.scatter(np.array(self.class0)[:,0], np.array(self.class0)[:,1], color='red')
                self.ax.scatter(np.array(self.class1)[:,0], np.array(self.class1)[:,1], color='blue')
                # Decision boundary
                x_vals = np.linspace(-5, 5, 200)
                w1, w2 = w[0,0], w[1,0]
                b = b[0]
                if w2 != 0:
                    y_vals = -(w1/w2) * x_vals - (b / w2)
                    self.ax.plot(x_vals, y_vals, color='black')
            else:  # Regression çizgisi
                x_vals = np.linspace(-5, 5, 200).reshape(-1,1)
                y_vals = x_vals @ w + b
                self.ax.plot(x_vals, y_vals, color='red')
                self.ax.scatter(X[:,0], y[:,0], color='green')
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.axhline(0, color='gray', linewidth=1)
            self.ax.axvline(0, color='gray', linewidth=1)
            self.ax.grid(True)
            self.ax.set_title(f"Epoch {epoch}, MSE={mse:.4f}")
            self.canvas.draw()

        self.status_label.config(text="⏳ Eğitim başlatıldı...")
        self.root.update()

        train_live(
            x_samples=X,
            y_true=y,
            Size=X.shape[0],
            Dim=X.shape[1],
            x_weights=weights,
            x_bias=bias,
            activation=activation,
            learning_const=learning_rate,
            Max_error=0.001,
            Max_epoch=max_epoch,
            on_update=update_line
        )

        self.status_label.config(text="✅ Eğitim tamamlandı!")

# --- Çalıştır ---
root = tk.Tk()
app = PerceptronGUI(root)
root.mainloop()
