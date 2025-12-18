import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.widgets import Button, RadioButtons, TextBox, Slider  # type: ignore
import time

# --- Veri saklama ---
class0 = []
class1 = []
current_class = [0]

# Eğitim parametreleri
epochs = [20]  

# --- Şekil / eksen ---
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.35)
ax.set_title("Click to add points (choose class at right).")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

scat0 = ax.scatter([], [], s=80, marker='o', color='blue', label='Class 0', edgecolors='black')
scat1 = ax.scatter([], [], s=80, marker='s', color='red', label='Class 1', edgecolors='black')
boundary_line, = ax.plot([], [], 'g-', linewidth=2, label='Decision boundary')

ax.legend(loc='upper left')

# --- Kontroller ---
rax = plt.axes([0.03, 0.65, 0.25, 0.15])
radio = RadioButtons(rax, ("Class 0", "Class 1"))
radio.on_clicked(lambda label: current_class.__setitem__(0, 0 if label=="Class 0" else 1))

train_ax = plt.axes([0.03, 0.55, 0.12, 0.06])
train_button = Button(train_ax, "Train")

# --- Reset butonu ---
reset_ax = plt.axes([0.17, 0.55, 0.12, 0.06])
reset_button = Button(reset_ax, "Reset")

def reset(event):
    class0.clear()
    class1.clear()
    # scatter sıfırlamak için bir nokta veriyoruz ama görünmez yapıyoruz
    scat0.set_offsets([[float('nan'), float('nan')]])
    scat1.set_offsets([[float('nan'), float('nan')]])

    boundary_line.set_data([], [])
    boundary_line.set_visible(False)
    fig.canvas.draw_idle()
    print("Data and decision boundary reset.")



reset_button.on_clicked(reset)


epoch_ax = plt.axes([0.03, 0.45, 0.15, 0.05])
epoch_box = TextBox(epoch_ax, "", initial=str(epochs[0]))
ax_epoch_label = fig.add_axes([0.03, 0.51, 0.15, 0.03])  # label için küçük ax
ax_epoch_label.axis("off")
ax_epoch_label.text(0, 0, "Epochs", fontsize=10)
epoch_box.on_submit(lambda text: epochs.__setitem__(0, max(1,int(text))))


# --- Mouse click event ---
def onclick(event):
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    if current_class[0] == 0:
        class0.append((x, y))
        scat0.set_offsets(class0)
    else:
        class1.append((x, y))
        scat1.set_offsets(class1)
    fig.canvas.draw_idle()

cid = fig.canvas.mpl_connect("button_press_event", onclick)

# --- Accuracy için text objesi (fig seviyesinde, eksenin üstünde kalır) ---
acc_text = fig.text(0.02, 0.95, "", fontsize=10,
                    verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

# --- Eğitim fonksiyonu ---
def train(event):
    boundary_line.set_visible(True)
    X = [pt for pt in class0] + [pt for pt in class1]
    labels = [0]*len(class0) + [1]*len(class1)
    if len(X) == 0:
        print("No points to train.")
        return

    w1 = w2 = b = 0.0
    eta = 0.1

    for epoch in range(epochs[0]):
        for i in range(len(X)):
            xi = X[i]
            net = w1*xi[0] + w2*xi[1] + b
            y_pred = 1 if net >= 0 else 0
            error = labels[i] - y_pred
            if error != 0:
                w1 += eta * error * xi[0]
                w2 += eta * error * xi[1]
                b  += eta * error

        # Her epoch'ta karar sınırını güncelle
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if abs(w2) < 1e-9:
            xval = -b / w1 if abs(w1) > 1e-9 else 0
            boundary_line.set_data([xval, xval], [ymin, ymax])
        else:
            xvals = [xmin, xmax]
            yvals = [-(w1*x + b)/w2 for x in xvals]
            boundary_line.set_data(xvals, yvals)

        fig.canvas.draw_idle()
        # plt.pause(0.1) - Event loop hatası veriyor, kaldırıldı

    # --- Training finished: accuracy hesapla ---
    correct = 0
    total_points = len(X)
    
    print(f"\n--- Accuracy Hesaplama Debug ---")
    print(f"Toplam nokta sayısı: {total_points}")
    print(f"Class 0 nokta sayısı: {len(class0)}")
    print(f"Class 1 nokta sayısı: {len(class1)}")
    print(f"Final ağırlıklar: w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}")
    
    for i, xi in enumerate(X):
        net = w1*xi[0] + w2*xi[1] + b
        y_pred = 1 if net >= 0 else 0
        actual = labels[i]
        is_correct = (y_pred == actual)
        
        if i < 5:  # İlk 5 noktanın detayını göster
            print(f"Nokta {i+1}: ({xi[0]:.2f}, {xi[1]:.2f}) -> net={net:.3f}, pred={y_pred}, actual={actual}, doğru={is_correct}")
        
        if is_correct:
            correct += 1
    
    accuracy = correct / total_points if total_points > 0 else 0
    
    print(f"Doğru tahmin: {correct}/{total_points}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("--- Debug Bitti ---\n")

    # Accuracy değerini grafiğe yaz
    acc_text.set_text(f"Accuracy: {accuracy*100:.2f}%")

    print(f"Training finished: w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}, accuracy={accuracy*100:.2f}%")


train_button.on_clicked(train)

plt.show()
