import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os, json, time

# ----------------------- Neural network (NumPy) -----------------------
class SimpleNN:
    def __init__(self, input_dim=2, hidden=12, output_dim=2, seed=None):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden) * 0.5
        self.b1 = np.zeros((1, hidden))
        self.W2 = rng.randn(hidden, output_dim) * 0.5
        self.b2 = np.zeros((1, output_dim))
        # adam-like not implemented, simple GD used

    def forward(self, X):
        z1 = X.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        return z1, a1, z2

    def predict_proba(self, X):
        _, _, z2 = self.forward(X)
        # softmax for probabilities
        ex = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

    def mse_loss(self, X, Y_onehot):
        _, _, z2 = self.forward(X)
        # compute MSE on raw outputs (to match user's MSE request)
        mse = np.mean((z2 - Y_onehot) ** 2)
        return mse

    def train(self, X, Y_onehot, epochs=100, lr=0.01, callback=None, stop_flag=None):
        # batch gradient descent
        n = X.shape[0]
        losses = []
        for ep in range(epochs):
            z1 = X.dot(self.W1) + self.b1
            a1 = np.tanh(z1)
            z2 = a1.dot(self.W2) + self.b2

            loss = np.mean((z2 - Y_onehot) ** 2)
            losses.append(loss)

            # gradients (MSE w.r.t z2 is 2*(z2 - Y)/n)
            dz2 = (2.0 / n) * (z2 - Y_onehot)  # shape n x C
            dW2 = a1.T.dot(dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2.dot(self.W2.T)
            dz1 = da1 * (1 - np.tanh(z1) ** 2)
            dW1 = X.T.dot(dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # gradient step
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

            if callback is not None:
                callback(ep, loss, losses)

            if stop_flag is not None and stop_flag():
                break

        return losses

# ----------------------- GUI Application -----------------------
class ANNVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('ANN Visualizer — Classification & MSE graph')
        self.geometry('1100x720')
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Data and model
        self.points = []  # list of (x,y,label,is_test)
        self.num_classes = tk.IntVar(value=2)
        self.selected_class = tk.IntVar(value=0)
        self.lr = tk.DoubleVar(value=0.01)
        self.epochs = tk.IntVar(value=100)
        self.hidden_size = tk.IntVar(value=12)
        self.net = None
        self.training = False
        self.stop_training = False
        self.mse_history = []

        self.create_widgets()
        self.draw_empty()

    def create_widgets(self):
        # left controls
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        ttk.Label(ctrl, text='Classes').pack(anchor='w')
        c_spin = ttk.Spinbox(ctrl, from_=2, to=10, textvariable=self.num_classes, width=5, command=self.on_classes_changed)
        c_spin.pack(anchor='w', pady=4)

        ttk.Label(ctrl, text='Selected class for adding:').pack(anchor='w', pady=(10,0))
        self.class_menu = ttk.Combobox(ctrl, values=list(range(self.num_classes.get())), textvariable=self.selected_class, width=5, state='readonly')
        self.class_menu.pack(anchor='w', pady=4)
        self.class_menu.current(0)

        ttk.Label(ctrl, text='Hidden neurons:').pack(anchor='w', pady=(10,0))
        ttk.Entry(ctrl, textvariable=self.hidden_size, width=7).pack(anchor='w', pady=4)

        ttk.Label(ctrl, text='Learning rate:').pack(anchor='w', pady=(10,0))
        ttk.Entry(ctrl, textvariable=self.lr, width=7).pack(anchor='w', pady=4)

        ttk.Label(ctrl, text='Epochs:').pack(anchor='w', pady=(10,0))
        ttk.Entry(ctrl, textvariable=self.epochs, width=7).pack(anchor='w', pady=4)

        ttk.Button(ctrl, text='Train', command=self.on_train).pack(fill='x', pady=(12,4))
        ttk.Button(ctrl, text='Stop Training', command=self.on_stop).pack(fill='x', pady=4)
        ttk.Button(ctrl, text='Add random class points', command=self.add_random).pack(fill='x', pady=4)
        ttk.Button(ctrl, text='Add test point (click plot)', command=self.enable_add_test).pack(fill='x', pady=4)
        ttk.Button(ctrl, text='Clear', command=self.clear_all).pack(fill='x', pady=4)

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=8)
        ttk.Button(ctrl, text='Save dataset', command=self.save_dataset).pack(fill='x', pady=2)
        ttk.Button(ctrl, text='Load dataset', command=self.load_dataset).pack(fill='x', pady=2)

        # matplotlib figure area
        fig = Figure(figsize=(7,6), dpi=100)
        self.ax_main = fig.add_axes([0.05, 0.35, 0.9, 0.6])
        self.ax_loss = fig.add_axes([0.05, 0.08, 0.9, 0.22])

        self.ax_main.set_xlim(-5,5)
        self.ax_main.set_ylim(-5,5)
        self.ax_main.set_title('2D coordinate plane — click to add points')
        self.ax_main.grid(True)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)

    # ----------------------- Data handling -----------------------
    def on_classes_changed(self):
        n = self.num_classes.get()
        self.class_menu['values'] = list(range(n))
        if self.selected_class.get() >= n:
            self.selected_class.set(0)
        self.draw_all()

    def add_point(self, x, y, label, is_test=False):
        self.points.append((x, y, int(label), bool(is_test)))
        self.draw_all()

    def add_random(self):
        n = self.num_classes.get()
        rng = np.random.RandomState(int(time.time()) % 100000)
        for cls in range(n):
            cx = rng.uniform(-3,3)
            cy = rng.uniform(-3,3)
            for i in range(6):
                x = cx + rng.randn()*0.4
                y = cy + rng.randn()*0.4
                self.add_point(x,y,cls,is_test=False)
        self.draw_all()

    def clear_all(self):
        self.points = []
        self.net = None
        self.mse_history = []
        self.draw_all()

    def save_dataset(self):
        file = filedialog.asksaveasfilename(defaultextension='.npz', filetypes=[('NPZ','*.npz')])
        if not file:
            return
        pts = np.array(self.points, dtype=object)
        np.savez(file, points=pts, num_classes=self.num_classes.get())
        messagebox.showinfo('Saved', f'Dataset saved to {file}')

    def load_dataset(self):
        file = filedialog.askopenfilename(filetypes=[('NPZ','*.npz')])
        if not file:
            return
        data = np.load(file, allow_pickle=True)
        pts = data['points']
        self.points = [tuple(p) for p in pts.tolist()]
        self.num_classes.set(int(data.get('num_classes',2)))
        self.on_classes_changed()
        self.draw_all()

    # ----------------------- Plotting -----------------------
    def draw_empty(self):
        self.ax_main.cla()
        self.ax_main.set_xlim(-5,5); self.ax_main.set_ylim(-5,5)
        self.ax_main.grid(True)
        self.ax_main.set_title('2D coordinate plane — click to add points')
        self.ax_loss.cla()
        self.ax_loss.set_title('MSE loss vs epochs')
        self.ax_loss.set_xlim(0,1); self.ax_loss.set_ylim(0,1)
        self.canvas.draw()

    def draw_all(self):
        self.ax_main.cla()
        self.ax_main.set_xlim(-5,5); self.ax_main.set_ylim(-5,5)
        self.ax_main.grid(True)
        self.ax_main.set_title('2D coordinate plane — click to add points')

        # draw decision boundary if trained
        if self.net is not None:
            xx = np.linspace(-5,5,200)
            yy = np.linspace(-5,5,200)
            XX, YY = np.meshgrid(xx,yy)
            grid = np.column_stack([XX.ravel(), YY.ravel()])
            probs = self.net.predict_proba(grid)
            pred = np.argmax(probs, axis=1).reshape(XX.shape)
            self.ax_main.contourf(XX, YY, pred, alpha=0.15, levels=np.arange(self.num_classes.get()+1)-0.5, cmap='tab10')

        # scatter points
        pts = np.array(self.points, dtype=float) if self.points else np.zeros((0,4))
        if pts.size:
            xs = pts[:,0].astype(float)
            ys = pts[:,1].astype(float)
            labels = pts[:,2].astype(int)
            is_test = pts[:,3].astype(bool)
            for cls in range(self.num_classes.get()):
                mask = labels==cls
                if mask.sum()>0:
                    self.ax_main.scatter(xs[mask & ~is_test], ys[mask & ~is_test], label=f'class {cls}', s=50)
                    self.ax_main.scatter(xs[mask & is_test], ys[mask & is_test], edgecolors='k', facecolors='none', marker='o', s=90, linewidths=1.5)

        self.ax_main.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
        # draw mse curve
        self.ax_loss.cla()
        self.ax_loss.set_title('MSE loss vs epochs')
        if len(self.mse_history)>0:
            self.ax_loss.plot(self.mse_history)
            self.ax_loss.set_xlim(0, max(1,len(self.mse_history)))
            self.ax_loss.set_ylim(0, max(1e-6, max(self.mse_history)*1.1))
        self.canvas.draw()

    # ----------------------- Interaction -----------------------
    def on_click(self, event):
        if event.inaxes != self.ax_main: 
            return
        x, y = event.xdata, event.ydata
        # if adding test mode enabled, add test point labeled -1 (unknown) for prediction
        if getattr(self, 'adding_test', False):
            self.add_point(x,y,label=self.selected_class.get(), is_test=True)
            self.adding_test = False
            self.draw_all()
            return
        # otherwise add training point with selected class
        cls = self.selected_class.get()
        self.add_point(x,y,cls, is_test=False)

    def enable_add_test(self):
        self.adding_test = True
        messagebox.showinfo('Add test point', 'Click on the plot to add a test point (marked with circle). It will be classified using current model when you Train or use Predict.')

    # ----------------------- Training -----------------------
    def on_train(self):
        if self.training:
            messagebox.showinfo('Training', 'Training already running.')
            return
        # prepare dataset
        pts = np.array(self.points, dtype=float)
        if pts.size==0:
            messagebox.showwarning('No data', 'No training points available. Add some data first.')
            return
        train_mask = pts[:,3]==0  # not test
        X = pts[train_mask,:2].astype(float)
        y = pts[train_mask,2].astype(int).astype(int)
        if X.shape[0] < self.num_classes.get():
            messagebox.showwarning('Data insufficient', 'Not enough points to train for all classes. Add more points.')
            return
        C = self.num_classes.get()
        Y_onehot = np.zeros((X.shape[0], C))
        Y_onehot[np.arange(X.shape[0]), y] = 1.0  # one-hot
        # Initialize network
        self.net = SimpleNN(input_dim=2, hidden=self.hidden_size.get(), output_dim=C, seed=1)
        self.mse_history = []
        self.training = True
        self.stop_training = False
        epochs = max(1, int(self.epochs.get()))
        lr = float(self.lr.get())

        def cb(epoch, loss, losses):
            # update MSE history every epoch and redraw
            self.mse_history = losses.copy()
            if epoch % 5 == 0 or epoch==epochs-1:
                self.draw_all()
            # process GUI events so window remains responsive
            self.update_idletasks()

        def stop_flag():
            return self.stop_training

        # run training in small chunks to keep UI responsive using after
        self.train_gen = self._train_generator(X, Y_onehot, epochs, lr, cb, stop_flag)
        self.after(1, self._train_step)

    def _train_generator(self, X, Y_onehot, epochs, lr, callback, stop_flag):
        # generator yields after each epoch
        for ep in range(epochs):
            losses = self.net.train(X, Y_onehot, epochs=1, lr=lr, callback=None, stop_flag=stop_flag)
            # train() returns list of losses for the epochs it ran (1 element)
            # we compute full MSE up to now
            if callback is not None:
                # compute full forward and mse to pass
                loss = self.net.mse_loss(X, Y_onehot)
                callback(ep, loss, getattr(self, 'mse_history', []) + [loss])
            yield ep
            if stop_flag():
                break

    def _train_step(self):
        try:
            next(self.train_gen)
            # append current loss
            pts = np.array(self.points, dtype=float)
            train_mask = pts[:,3]==0 if pts.size else np.array([],dtype=bool)
            if pts.size:
                X = pts[train_mask,:2].astype(float)
                y = pts[train_mask,2].astype(int).astype(int)
                C = self.num_classes.get()
                Y_onehot = np.zeros((X.shape[0], C))
                Y_onehot[np.arange(X.shape[0]), y] = 1.0
                self.mse_history.append(self.net.mse_loss(X, Y_onehot))
            self.draw_all()
            self.update_idletasks()
            if not self.stop_training:
                self.after(10, self._train_step)
            else:
                self.training = False
        except StopIteration:
            self.training = False
            self.stop_training = False
            self.draw_all()
            messagebox.showinfo('Finished', 'Training finished. Decision boundary updated.')

    def on_stop(self):
        self.stop_training = True

    # ----------------------- Save/Load Model or Dataset helpers -----------------------
    def on_close(self):
        if self.training and not messagebox.askyesno('Exit', 'Training running. Exit anyway?'):
            return
        self.destroy()

# ----------------------- Run app -----------------------
if __name__ == '__main__':
    app = ANNVisualizer()
    app.mainloop()
