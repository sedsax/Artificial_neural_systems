import matplotlib.pyplot as plt
import re

# Loss değerlerini log dosyasından veya çıktıdan çekmek için bir fonksiyon
def extract_losses_from_log(log_path):
    losses = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r"Loss: ([0-9.]+)", line)
            if match:
                losses.append(float(match.group(1)))
    return losses

if __name__ == "__main__":
    # Kendi çıktı dosyanızın adını buraya yazın
    log_file = "mnist_train_output.txt"  # Örneğin: python .\mnist_train.py > mnist_train_output.txt
    losses = extract_losses_from_log(log_file)
    if not losses:
        print("Loss verisi bulunamadı. Lütfen doğru log dosyasını belirtin.")
    else:
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Eğitim Kayıp (Loss) Grafiği')
        plt.grid(True)
        plt.show()
