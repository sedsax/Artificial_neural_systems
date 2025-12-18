import matplotlib.pyplot as plt

# 50 epoch boyunca elde edilen loss değerleri (örnek olarak verdiğiniz verilerden derlenmiştir)
losses = [
    1.9872, 1.3396, 0.9425, 0.7638, 0.6547, 0.5797, 0.5264, 0.4880, 0.4601, 0.4400,
    0.4257, 0.4149, 0.4055, 0.3962, 0.3863, 0.3759, 0.3653, 0.3548, 0.3448, 0.3353,
    0.3263, 0.3180, 0.3101, 0.3027, 0.2956, 0.2885, 0.2814, 0.2741, 0.2664, 0.2582,
    0.2497, 0.2408, 0.2317, 0.2224, 0.2131, 0.2039, 0.1949, 0.1862, 0.1778, 0.1699,
    0.1623, 0.1552, 0.1486, 0.1426, 0.1371, 0.1323, 0.1281, 0.1247, 0.1220, 0.1198
]

plt.figure(figsize=(8,5))
plt.plot(range(1, len(losses)+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Eğitim Kayıp (Loss) Grafiği')
plt.grid(True)
plt.tight_layout()
plt.show()
