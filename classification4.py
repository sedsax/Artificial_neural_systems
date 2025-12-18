import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

max_classes = 10
num_classes = [3]  # mutable, textbox ile değişecek
classes = [[] for _ in range(max_classes)]
current_class = [0]
epochs = [20]

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.35)
ax.set_title("Click to add points (choose class at right).")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '8']

scatters = []
boundary_lines = []

def setup_class_controls():
    global scatters, boundary_lines, radio
    # Temizle
    for coll in ax.collections:
        coll.remove()
    for line in ax.lines:
        line.remove()
    scatters = [
        ax.scatter([], [], s=80, marker=markers[i], color=colors[i], label=f'Class {i}', edgecolors='black')
        for i in range(num_classes[0])
    ]
    boundary_lines = [
        ax.plot([], [], color=colors[i], linewidth=2, label=f'Boundary {i}')[0]
        for i in range(num_classes[0])
    ]
    ax.legend(loc='upper left')
    # Radio buttons
    rax.clear()
    radio.labels = []
    radio.value_selected = f"Class 0"
    radio.__init__(rax, [f"Class {i}" for i in range(num_classes[0])])
    radio.on_clicked(lambda label: current_class.__setitem__(0, int(label.split()[-1])))
    fig.canvas.draw_idle()

# --- Kontroller ---
rax = plt.axes([0.03, 0.65, 0.25, 0.15])
radio = RadioButtons(rax, [f"Class {i}" for i in range(num_classes[0])])
radio.on_clicked(lambda label: current_class.__setitem__(0, int(label.split()[-1])))

train_ax = plt.axes([0.03, 0.55, 0.12, 0.06])
train_button = Button(train_ax, "Train")

reset_ax = plt.axes([0.17, 0.55, 0.12, 0.06])
reset_button = Button(reset_ax, "Reset")

def reset(event):
    for i in range(max_classes):
        classes[i].clear()
    for scat in scatters:
        scat.set_offsets([[float('nan'), float('nan')]])
    for line in boundary_lines:
        line.set_data([], [])
        line.set_visible(False)
    fig.canvas.draw_idle()

reset_button.on_clicked(reset)

epoch_ax = plt.axes([0.03, 0.45, 0.15, 0.05])
epoch_box = TextBox(epoch_ax, "", initial=str(epochs[0]))
ax_epoch_label = fig.add_axes([0.03, 0.51, 0.15, 0.03])
ax_epoch_label.axis("off")
ax_epoch_label.text(0, 0, "Epochs", fontsize=10)
epoch_box.on_submit(lambda text: epochs.__setitem__(0, max(1,int(text))))

# Sınıf sayısı kutusu
classnum_ax = plt.axes([0.03, 0.35, 0.15, 0.05])
classnum_box = TextBox(classnum_ax, "Class count", initial=str(num_classes[0]))
def update_classnum(text):
    val = max(2, min(max_classes, int(text)))
    num_classes[0] = val
    setup_class_controls()
classnum_box.on_submit(update_classnum)

def onclick(event):
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    idx = current_class[0]
    classes[idx].append((x, y))
    scatters[idx].set_offsets(classes[idx])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", onclick)

acc_text = fig.text(0.02, 0.95, "", fontsize=10,
                    verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

def train(event):
    for line in boundary_lines:
        line.set_visible(True)
    X = [pt for i in range(num_classes[0]) for pt in classes[i]]
    labels = []
    for i in range(num_classes[0]):
        labels += [i]*len(classes[i])
    if len(X) == 0:
        print("No points to train.")
        return

    # One-vs-rest perceptron
    weights = []
    for k in range(num_classes[0]):
        w1 = w2 = b = 0.0
        eta = 0.1
        yk = [1 if lbl==k else 0 for lbl in labels]
        for epoch in range(epochs[0]):
            for i in range(len(X)):
                xi = X[i]
                net = w1*xi[0] + w2*xi[1] + b
                y_pred = 1 if net >= 0 else 0
                error = yk[i] - y_pred
                if error != 0:
                    w1 += eta * error * xi[0]
                    w2 += eta * error * xi[1]
                    b  += eta * error
        weights.append((w1, w2, b))

    # Karar sınırlarını çiz
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for k, (w1, w2, b) in enumerate(weights):
        if abs(w2) < 1e-9:
            xval = -b / w1 if abs(w1) > 1e-9 else 0
            boundary_lines[k].set_data([xval, xval], [ymin, ymax])
        else:
            xvals = [xmin, xmax]
            yvals = [-(w1*x + b)/w2 for x in xvals]
            boundary_lines[k].set_data(xvals, yvals)
    fig.canvas.draw_idle()

    # Accuracy hesapla (en yüksek net ile tahmin)
    correct = 0
    for i, xi in enumerate(X):
        nets = [w1*xi[0] + w2*xi[1] + b for (w1, w2, b) in weights]
        pred = nets.index(max(nets))
        if pred == labels[i]:
            correct += 1
    accuracy = correct / len(X)
    acc_text.set_text(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Training finished. Accuracy={accuracy*100:.2f}%")

train_button.on_clicked(train)

setup_class_controls()
plt.show()