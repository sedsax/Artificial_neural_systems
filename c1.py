import math
import time

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.widgets import Button, RadioButtons, TextBox, Slider  # type: ignore

# ----------------- Global state -----------------
# Data containers
class0 = []          # list[(x, y)]
class1 = []          # list[(x, y)]
reg_points = []      # list[(x, y)]  -> regression uses y ≈ w*x + b

# Test click containers (viz only, not part of training)
test_bin_0 = []      # predicted class 0 test points
test_bin_1 = []      # predicted class 1 test points
test_reg_pred = []   # list[(x, y_pred)] markers for regression test clicks

# UI state
mode = ["Binary"]          # "Binary" or "Regression"
click_action = ["Add"]     # "Add" or "Test"
current_class = [0]        # for Binary: 0 or 1
epochs = [200]
lr = [0.1]

# Models
bin_model = {"w1": 0.0, "w2": 0.0, "b": 0.0, "trained": False}
reg_model = {"w": 0.0, "b": 0.0, "trained": False}

# Histories
mse_history = []     # per-epoch MSE

# Plot handles
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.35, bottom=0.3)  # leave space for MSE plot and controls
ax.set_title("Click to add points")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# MSE axes
ax_mse = fig.add_axes([0.35, 0.05, 0.6, 0.2])
ax_mse.set_title("MSE per epoch")
ax_mse.set_xlabel("epoch")
ax_mse.set_ylabel("MSE")
mse_line, = ax_mse.plot([], [], color="black")

# Background heatmap (classification)
heatmap_im = None

# Scatter plots
scat0 = ax.scatter([], [], s=80, marker='o', color='blue', label='Class 0', edgecolors='black')
scat1 = ax.scatter([], [], s=80, marker='s', color='red', label='Class 1', edgecolors='black')
# test prediction scatters (hollow)
test0_scat = ax.scatter([], [], s=120, marker='o', facecolors='none', edgecolors='blue', linewidths=2, label='Pred 0')
test1_scat = ax.scatter([], [], s=120, marker='o', facecolors='none', edgecolors='red', linewidths=2, label='Pred 1')

# Regression scatters/line
scat_reg = ax.scatter([], [], s=80, marker='o', color='purple', edgecolors='black', label='Reg data')
reg_pred_scat = ax.scatter([], [], s=80, marker='x', color='black', linewidths=2, label='y_pred (test)')
reg_line, = ax.plot([], [], 'g-', linewidth=2, label='Regression line')

# Decision boundary line (classification)
boundary_line, = ax.plot([], [], 'g-', linewidth=2, label='Decision boundary')

ax.legend(loc='upper left')

# Status text
status_text = fig.text(0.02, 0.96, "", fontsize=10,
                       verticalalignment="top",
                       bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

def set_status(msg):
    status_text.set_text(msg)
    fig.canvas.draw_idle()

# ----------------- Math helpers -----------------
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

# ----------------- Training loops (shared patterns) -----------------
def train_binary_mse(X, y, n_epochs, learning_rate):
    # X: list[(x, y)], y: list[0.0/1.0]
    w1 = bin_model["w1"]
    w2 = bin_model["w2"]
    b  = bin_model["b"]
    hist = []
    N = max(1, len(X))
    for ep in range(n_epochs):
        sse = 0.0
        # Simple SGD (fixed order)
        for i in range(len(X)):
            x1, x2 = X[i]
            t = y[i]
            z = w1*x1 + w2*x2 + b
            a = sigmoid(z)
            err = a - t
            sse += err * err
            # dMSE/dz = 2*(a - t) * a*(1-a)
            dL_dz = 2.0 * err * a * (1.0 - a)
            w1 -= learning_rate * dL_dz * x1
            w2 -= learning_rate * dL_dz * x2
            b  -= learning_rate * dL_dz
        hist.append(sse / N)
    return w1, w2, b, hist

def train_regression_mse(X, y, n_epochs, learning_rate):
    # 1D linear regression: y ≈ w*x + b, using clicked (x,y)
    w = reg_model["w"]
    b = reg_model["b"]
    hist = []
    N = max(1, len(X))
    for ep in range(n_epochs):
        sse = 0.0
        for i in range(len(X)):
            x = X[i]
            t = y[i]
            yhat = w*x + b
            err = yhat - t
            sse += err * err
            # gradients
            dw = 2.0 * err * x
            db = 2.0 * err
            w -= learning_rate * dw
            b -= learning_rate * db
        hist.append(sse / N)
    return w, b, hist

# ----------------- Drawing helpers -----------------
def update_scatters():
    # training data
    if class0:
        scat0.set_offsets(class0)
    else:
        scat0.set_offsets([[float('nan'), float('nan')]])
    if class1:
        scat1.set_offsets(class1)
    else:
        scat1.set_offsets([[float('nan'), float('nan')]])
    # regression data
    if reg_points:
        scat_reg.set_offsets(reg_points)
    else:
        scat_reg.set_offsets([[float('nan'), float('nan')]])
    # test scatters
    if test_bin_0:
        test0_scat.set_offsets(test_bin_0)
    else:
        test0_scat.set_offsets([[float('nan'), float('nan')]])
    if test_bin_1:
        test1_scat.set_offsets(test_bin_1)
    else:
        test1_scat.set_offsets([[float('nan'), float('nan')]])
    if test_reg_pred:
        reg_pred_scat.set_offsets(test_reg_pred)
    else:
        reg_pred_scat.set_offsets([[float('nan'), float('nan')]])
    fig.canvas.draw_idle()

def draw_decision_boundary():
    # Draw line w1*x + w2*y + b = 0
    if not bin_model["trained"]:
        boundary_line.set_data([], [])
        return
    w1, w2, b = bin_model["w1"], bin_model["w2"], bin_model["b"]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if abs(w2) < 1e-12:
        # vertical line x = -b/w1
        if abs(w1) < 1e-12:
            boundary_line.set_data([], [])
            return
        xval = -b / w1
        boundary_line.set_data([xval, xval], [ymin, ymax])
    else:
        xvals = [xmin, xmax]
        yvals = [-(w1*x + b)/w2 for x in xvals]
        boundary_line.set_data(xvals, yvals)
    fig.canvas.draw_idle()

def draw_regression_line():
    if not reg_model["trained"]:
        reg_line.set_data([], [])
        return
    w, b = reg_model["w"], reg_model["b"]
    xmin, xmax = ax.get_xlim()
    xvals = [xmin, xmax]
    yvals = [w*x + b for x in xvals]
    reg_line.set_data(xvals, yvals)
    fig.canvas.draw_idle()

def plot_mse_history():
    ax_mse.clear()
    ax_mse.set_title("MSE per epoch")
    ax_mse.set_xlabel("epoch")
    ax_mse.set_ylabel("MSE")
    if mse_history:
        xs = list(range(1, len(mse_history) + 1))
        ax_mse.plot(xs, mse_history, color="black")
    fig.canvas.draw_idle()

def clear_heatmap():
    global heatmap_im
    if heatmap_im is not None:
        heatmap_im.remove()
        heatmap_im = None

def show_map_classification():
    # colored regions based on prediction
    global heatmap_im
    if not bin_model["trained"]:
        set_status("Train (Binary) first.")
        return
    Xmin, Xmax = ax.get_xlim()
    Ymin, Ymax = ax.get_ylim()
    res = 180
    dx = (Xmax - Xmin) / (res - 1)
    dy = (Ymax - Ymin) / (res - 1)
    grid = []
    for gy in range(res):
        row = []
        yv = Ymin + gy * dy
        for gx in range(res):
            xv = Xmin + gx * dx
            z = bin_model["w1"]*xv + bin_model["w2"]*yv + bin_model["b"]
            a = sigmoid(z)
            lbl = 1 if a >= 0.5 else 0
            row.append(lbl)
        grid.append(row)
    if heatmap_im is not None:
        heatmap_im.remove()
    cmap = matplotlib.colors.ListedColormap([
        (0.1, 0.6, 0.9, 0.20),  # semi-transparent blue
        (0.9, 0.2, 0.2, 0.20)   # semi-transparent red
    ])
    heatmap_im = ax.imshow(grid, extent=[Xmin, Xmax, Ymin, Ymax],
                           origin="lower", cmap=cmap, interpolation="nearest")
    fig.canvas.draw_idle()

# ----------------- Controls -----------------
# Mode
ax_mode = plt.axes([0.03, 0.86, 0.25, 0.10])
rb_mode = RadioButtons(ax_mode, ("Binary", "Regression"))
def on_mode(label):
    mode[0] = label
    # Toggle visibility
    ax_class.set_visible(mode[0] == "Binary")
    ax_click.set_visible(True)
    # Clear overlays irrelevant to mode
    clear_heatmap()
    boundary_line.set_data([], [])
    reg_line.set_data([], [])
    # Keep train/test clicks separate per mode
    fig.canvas.draw_idle()
rb_mode.on_clicked(on_mode)

# Click Action
ax_click = plt.axes([0.03, 0.75, 0.25, 0.10])
rb_click = RadioButtons(ax_click, ("Add", "Test"))
def on_click_act(label):
    click_action[0] = label
rb_click.on_clicked(on_click_act)

# Class selector (binary)
ax_class = plt.axes([0.03, 0.64, 0.25, 0.10])
rb_class = RadioButtons(ax_class, ("Class 0", "Class 1"))
def on_class(label):
    current_class[0] = 0 if label == "Class 0" else 1
rb_class.on_clicked(on_class)

# Epochs
ax_epoch_lab = fig.add_axes([0.03, 0.59, 0.12, 0.03]); ax_epoch_lab.axis("off")
ax_epoch_lab.text(0, 0, "Epochs", fontsize=10)
ax_epoch = plt.axes([0.15, 0.58, 0.13, 0.05])
tb_epoch = TextBox(ax_epoch, "", initial=str(epochs[0]))
def on_epoch(text):
    try:
        epochs[0] = max(1, int(float(text)))
    except:
        pass
tb_epoch.on_submit(on_epoch)

# Learning rate
ax_lr_lab = fig.add_axes([0.03, 0.53, 0.12, 0.03]); ax_lr_lab.axis("off")
ax_lr_lab.text(0, 0, "lr", fontsize=10)
ax_lr_box = plt.axes([0.15, 0.52, 0.13, 0.05])
tb_lr = TextBox(ax_lr_box, "", initial=str(lr[0]))
def on_lr(text):
    try:
        lr[0] = float(text)
    except:
        pass
tb_lr.on_submit(on_lr)

# Train
ax_train = plt.axes([0.03, 0.45, 0.12, 0.06])
btn_train = Button(ax_train, "Train")

# Show Map
ax_map = plt.axes([0.16, 0.45, 0.12, 0.06])
btn_map = Button(ax_map, "Show Map")

# Reset
ax_reset = plt.axes([0.03, 0.37, 0.25, 0.06])
btn_reset = Button(ax_reset, "Reset")

# ----------------- Events -----------------
def onclick(event):
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    x, y = float(event.xdata), float(event.ydata)

    if mode[0] == "Binary":
        if click_action[0] == "Add":
            if current_class[0] == 0:
                class0.append((x, y))
            else:
                class1.append((x, y))
        else:  # Test
            if not bin_model["trained"]:
                set_status("Train (Binary) first to test.")
            else:
                z = bin_model["w1"]*x + bin_model["w2"]*y + bin_model["b"]
                a = sigmoid(z)
                pred = 1 if a >= 0.5 else 0
                if pred == 0:
                    test_bin_0.append((x, y))
                else:
                    test_bin_1.append((x, y))
    else:  # Regression
        if click_action[0] == "Add":
            reg_points.append((x, y))
        else:  # Test
            if not reg_model["trained"]:
                set_status("Train (Regression) first to test.")
            else:
                y_pred = reg_model["w"]*x + reg_model["b"]
                test_reg_pred.append((x, y_pred))

    update_scatters()

cid = fig.canvas.mpl_connect("button_press_event", onclick)

def train(event):
    global mse_history
    clear_heatmap()

    if mode[0] == "Binary":
        X = [pt for pt in class0] + [pt for pt in class1]
        y = [0.0]*len(class0) + [1.0]*len(class1)
        if len(class0) == 0 or len(class1) == 0:
            set_status("Binary: add points for BOTH classes.")
            return
        start = time.time()
        w1, w2, b, mse_history = train_binary_mse(X, y, epochs[0], lr[0])
        elapsed = time.time() - start
        bin_model["w1"] = w1
        bin_model["w2"] = w2
        bin_model["b"]  = b
        bin_model["trained"] = True

        # Train accuracy
        correct = 0
        for i in range(len(X)):
            x1, x2 = X[i]
            t = y[i]
            a = sigmoid(w1*x1 + w2*x2 + b)
            pred = 1.0 if a >= 0.5 else 0.0
            if pred == t:
                correct += 1
        acc = correct / len(X)

        plot_mse_history()
        draw_decision_boundary()
        reg_line.set_data([], [])  # hide reg line in binary
        set_status(f"Trained (Binary) in {elapsed:.2f}s | final MSE={mse_history[-1]:.4f} | Acc={acc*100:.2f}%")

    else:  # Regression
        if not reg_points:
            set_status("Regression: add points first.")
            return
        X = [p[0] for p in reg_points]
        y = [p[1] for p in reg_points]
        start = time.time()
        w, b, mse_history = train_regression_mse(X, y, epochs[0], lr[0])
        elapsed = time.time() - start
        reg_model["w"] = w
        reg_model["b"] = b
        reg_model["trained"] = True

        plot_mse_history()
        draw_regression_line()
        boundary_line.set_data([], [])  # hide decision boundary in regression
        set_status(f"Trained (Regression) in {elapsed:.2f}s | final MSE={mse_history[-1]:.4f}")

btn_train.on_clicked(train)

def show_map(event):
    if mode[0] == "Binary":
        show_map_classification()
        draw_decision_boundary()  # keep boundary visible on top
    else:
        clear_heatmap()
        draw_regression_line()
        set_status("Regression map: showing fitted line.")
btn_map.on_clicked(show_map)

def reset(event):
    class0.clear()
    class1.clear()
    reg_points.clear()
    test_bin_0.clear()
    test_bin_1.clear()
    test_reg_pred.clear()
    mse_history.clear()

    bin_model["w1"] = 0.0; bin_model["w2"] = 0.0; bin_model["b"] = 0.0; bin_model["trained"] = False
    reg_model["w"] = 0.0;  reg_model["b"]  = 0.0;  reg_model["trained"]  = False

    boundary_line.set_data([], [])
    reg_line.set_data([], [])
    clear_heatmap()
    update_scatters()
    plot_mse_history()
    set_status("Reset.")
btn_reset.on_clicked(reset)

# ----------------- Init -----------------
update_scatters()
plot_mse_history()
set_status("Mode: Binary. Click Action: Add. Add points and press Train.")
plt.show()