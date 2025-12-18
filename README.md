w₁x + w₂y + b = 0
w = w + η * (y - ŷ) * x

# Flexible Artificial Neural Network Visualizer

Bu proje, çok katmanlı yapay sinir ağlarının (ANN) saf Python ile görsel ve etkileşimli olarak eğitilmesini ve test edilmesini sağlar. Sınıflandırma ve regresyon problemleri için kullanılabilir.

## Özellikler

- **Görsel Veri Girişi**: Mouse ile 2D uzayda nokta ekleme
- **Çok Katmanlı ANN**: İstenilen sayıda gizli katman ve nöron
- **Sınıflandırma & Regresyon**: Her iki problem tipi için destek
- **Momentum Seçeneği**: Momentum ile veya momentum olmadan eğitim
- **Parametre Kontrolü**: Katman sayısı, nöron sayısı, öğrenme oranı, momentum, epoch
- **Gerçek Zamanlı Görselleştirme**: Sınır çizgisi (decision boundary) ve loss grafiği
- **Saf Python**: NumPy kullanılmaz, tüm matris işlemleri Python ile
- **Matplotlib ile boundary çizimi**

## Nasıl Çalışır?

1. **Problem Tipi Seçimi**: Sınıflandırma veya regresyon seçin
2. **Veri Girişi**: Mouse ile noktalar ekleyin, sınıf seçimini değiştirerek farklı sınıflar ekleyin
3. **Ağ Parametreleri**: Katman, nöron, öğrenme oranı, momentum gibi parametreleri ayarlayın
4. **Eğitim**: "Train" butonuna basarak ağı eğitin
5. **Sonuç**: Sınır çizgisi ve loss grafiği otomatik olarak güncellenir

## Kullanım

```bash
python ann_project.py
```

## Gereksinimler

- Python 3.7+
- Matplotlib
- Tkinter (Python ile birlikte gelir)

**Not:** Proje tamamen saf Python ile yazılmıştır. NumPy, scikit-learn gibi kütüphaneler kullanılmaz. Sadece görselleştirme için matplotlib kullanılır.

## Algoritma Detayları

- **İleri Yayılım (Forward Propagation)**: Her katmanda tanh aktivasyonu (son katmanda softmax veya linear)
- **Geri Yayılım (Backpropagation)**: Cross-entropy loss (sınıflandırma) veya MSE (regresyon), momentum destekli ağırlık güncelleme
- **Momentum**: İsteğe bağlı olarak açılıp kapatılabilir

### Örnek Eğitim Döngüsü
```python
net = FlexibleANN([2, 8, 4, 2], lr=0.01, momentum=0.9)
losses = net.train(X, y, epochs=100)
```

### Sınıflandırma ve Regresyon
- Sınıflandırmada softmax çıkış, regresyonda linear çıkış kullanılır.

## Proje Dosyası: ann_project.py

Tüm kod ve görsel arayüz tek dosyada bulunur. Matris işlemleri, aktivasyonlar, eğitim döngüsü ve görselleştirme fonksiyonları ayrıntılı olarak Python ile yazılmıştır.

---

## Detaylı Teknik Rapor ve Dökümantasyon

### Genel Bakış
Bu proje, saf Python ile yazılmış, çok katmanlı yapay sinir ağlarının (ANN) hem sınıflandırma hem de regresyon problemleri için görsel ve etkileşimli olarak eğitilebildiği bir uygulamadır. Kullanıcı, arayüz üzerinden veri noktaları ekleyebilir, ağ mimarisini ve eğitim parametrelerini değiştirebilir, eğitim sürecini başlatabilir ve sonuçları anında görselleştirebilir.

### Temel Bileşenler

- **Matris İşlemleri:** Tüm matris çarpma, toplama, transpoz, aktivasyon ve türev işlemleri saf Python ile fonksiyonlar halinde yazılmıştır.
- **Aktivasyon ve Kayıp Fonksiyonları:** Softmax (sınıflandırma), tanh (gizli katman), cross-entropy loss (sınıflandırma), linear çıkış (regresyon).
- **FlexibleANN Sınıfı:** Ağ mimarisi, ileri yayılım, geri yayılım, momentumlu ağırlık güncelleme, tahmin ve eğitim fonksiyonlarını içerir.
- **Görsel Arayüz:** Tkinter tabanlı, kullanıcı dostu bir GUI. Veri girişi, parametre ayarı, eğitim ve test işlemleri, boundary ve loss grafiği.
- **Decision Boundary ve Loss Grafiği:** Matplotlib ile 2D uzayda modelin karar sınırı ve loss grafiği çizilir.

### Eğitim (Backpropagation) Adımları
1. İleri yayılım ile tüm katman aktivasyonları hesaplanır.
2. Çıkış ile hedef arasındaki hata (delta) bulunur.
3. Her katmanda ağırlık ve bias gradyanları hesaplanır.
4. Momentumlu güncelleme ile ağırlıklar ve biaslar güncellenir.
5. Delta, bir önceki katmana geri yayılır.

### Kullanım Senaryosu
1. `python ann_project.py` ile başlatılır.
2. Mouse ile noktalar eklenir, sınıf seçimi yapılır.
3. Katman, nöron, öğrenme oranı, momentum gibi parametreler ayarlanır.
4. Train butonuna basılır, model eğitilir ve sonuçlar görselleştirilir.
5. Yeni bir nokta eklenip modelin tahmini gözlemlenir.
6. Temizle ile veri ve model sıfırlanır.

### Teknik Notlar
- Momentum: Ağırlık güncellemelerinde önceki adımın etkisi de eklenir.
- Katman ve Nöron Sayısı: Kullanıcı arayüzünden istenildiği gibi değiştirilebilir.
- Regresyon Modu: Son katmanda aktivasyon uygulanmaz, loss fonksiyonu MSE olur.
- Sınıflandırma Modu: Son katmanda softmax, loss fonksiyonu cross-entropy olur.
- Matplotlib: Sadece boundary ve loss grafiği için kullanılır.

### Koddan Örnekler
```python
net = FlexibleANN([2, 8, 4, 2], lr=0.01, momentum=0.9)
losses = net.train(X, y, epochs=100)
```

### Güçlü ve Zayıf Yönler
**Güçlü Yönler:**
- Tamamen saf Python, eğitim amaçlı şeffaf kod
- Görsel ve etkileşimli arayüz
- Momentum, çok katman, parametrik yapı

**Zayıf Yönler:**
- NumPy kullanılmadığı için büyük veriyle yavaş çalışır
- Sadece 2D veri ve görselleştirme için uygundur
- Mini-batch veya epoch başına shuffle yok (isteğe göre eklenebilir)

### Sonuç
Bu proje, yapay sinir ağlarının temel prensiplerini, ileri/geri yayılımı ve momentumlu öğrenmeyi görsel olarak anlamak ve denemek için ideal bir eğitim aracıdır. Kodun tamamı şeffaf ve özelleştirilebilir şekilde yazılmıştır.