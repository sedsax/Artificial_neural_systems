import sys
import os

# Modülleri import et
from modules.core_ann import FlexibleANN, mat_mul, mat_add_bias, transpose, subtract_matrix, tanh_matrix, tanh_derivative_matrix, softmax_matrix, cross_entropy, argmax_row
from modules.mnist_utils import load_mnist_images, load_mnist_labels, one_hot
from modules.feature_extraction import extract_features
from modules.visualizer import ANNVisualizer, MNISTDrawWindow

import tkinter as tk

# 2D ANN (ann_project) fonksiyonu
def run_ann_project():
    root = tk.Tk()
    # ANNVisualizer'a gerekli fonksiyonları parametre olarak veriyoruz
    ANNVisualizer(root, FlexibleANN, mat_mul, transpose, tanh_derivative_matrix, cross_entropy, subtract_matrix, softmax_matrix, argmax_row)
    root.mainloop()

# MNIST MLP fonksiyonu
def run_mnist():
    def train_mnist_model():
        import time
        start = time.time()
        X_train = load_mnist_images("train-images.idx3-ubyte", limit=2000)
        y_train_lbl = load_mnist_labels("train-labels.idx1-ubyte", limit=2000)
        y_train = one_hot(y_train_lbl)
        X_test = load_mnist_images("t10k-images.idx3-ubyte", limit=200)
        y_test_lbl = load_mnist_labels("t10k-labels.idx1-ubyte", limit=200)
        net = FlexibleANN([784, 64, 32, 10], lr=0.01, momentum=0.9)
        net.train(X_train, y_train, epochs=50, batch_size=8)
        acc = sum([net.predict(x) == y for x, y in zip(X_test, y_test_lbl)]) / len(X_test)
        print("\nTest Accuracy:", round(acc*100, 2), "%")
        print("Elapsed time:", round(time.time()-start, 2), "seconds")
        net.save_weights("mnist_ann_weights.pkl")
        return net
    root = tk.Tk()
    weights_file = "mnist_ann_weights.pkl"
    if os.path.exists(weights_file):
        print("Ağırlık dosyası bulundu, yükleniyor...")
        net = FlexibleANN([784, 64, 32, 10], lr=0.01, momentum=0.9)
        net.load_weights(weights_file)
    else:
        net = train_mnist_model()
    root.title("ANN Visualizer + MNIST")
    tk.Button(root, text="Draw MNIST Digit", command=lambda: MNISTDrawWindow(net)).pack(pady=20)
    root.mainloop()

# MNIST + CNN feature extraction fonksiyonu
def run_cnn():
    import random
    import time
    kernel = [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(3)]
    start = time.time()
    X_train_raw = load_mnist_images("train-images.idx3-ubyte", limit=2000)
    y_train_lbl = load_mnist_labels("train-labels.idx1-ubyte", limit=2000)
    y_train = one_hot(y_train_lbl)
    X_train = [extract_features(x, kernel) for x in X_train_raw]
    X_test_raw = load_mnist_images("t10k-images.idx3-ubyte", limit=200)
    y_test_lbl = load_mnist_labels("t10k-labels.idx1-ubyte", limit=200)
    X_test = [extract_features(x, kernel) for x in X_test_raw]
    net = FlexibleANN([169, 64, 32, 10], lr=0.01, momentum=0.9)
    net.train(X_train, y_train, epochs=50, batch_size=8)
    acc = sum([net.predict(x) == y for x, y in zip(X_test, y_test_lbl)]) / len(X_test)
    print("\nTest Accuracy:", round(acc*100, 2), "%")
    print("Elapsed time:", round(time.time()-start, 2), "seconds")

if __name__ == "__main__":
    print("Seçenekler:")
    print("1 - 2D ANN (ann_project)")
    print("2 - MNIST MLP (mnist_train)")
    print("3 - MNIST + CNN Feature Extraction (cnn_train)")
    secim = input("Çalıştırmak istediğiniz projeyi seçin (1/2/3): ")
    if secim == "1":
        run_ann_project()
    elif secim == "2":
        run_mnist()
    elif secim == "3":
        run_cnn()
    else:
        print("Geçersiz seçim.")
