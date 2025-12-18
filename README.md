# Linear Classifier - Artificial Neural Systems

Bu proje, doğrusal sınıflandırma algoritmasının (Perceptron) görsel bir implementasyonudur. 

## Özellikler

- **Görsel Veri Girişi**: Mouse ile koordinat sisteminde nokta işaretleme
- **İki Sınıf Desteği**: Class 0 (mavi) ve Class 1 (kırmızı)
- **Perceptron Algoritması**: Saf Python implementasyonu
- **Real-time Görselleştirme**: Decision boundary'nin çizilmesi
- **Parametre Kontrolü**: Epoch sayısı ve learning rate ayarları

## Nasıl Çalışır?

1. **Veri Toplama**: Dropdown'dan bir sınıf seçin ve mouse ile noktalar işaretleyin
2. **Sınıf Değiştirme**: Diğer sınıfı seçip farklı bölgelere noktalar ekleyin  
3. **Eğitim**: "TRAIN" butonuna basarak modeli eğitin
4. **Sonuç**: Yeşil kesikli çizgi decision boundary'yi gösterir

## Matematiksel Temel

Perceptron algoritması şu formülü öğrenir:
```
w₁x + w₂y + b = 0
```

Bu doğru iki sınıfı ayıran karar sınırını temsil eder.

## Kullanım

```bash
python linear_classifier.py
```

## Gereksinimler

- Python 3.7+
- Matplotlib (sadece görselleştirme için)
- Tkinter (Python ile gelir)

**ÖNEMLİ**: Bu proje tamamen saf Python ile yazılmıştır. NumPy, math, random gibi kütüphaneler kullanılmamıştır. Sadece görselleştirme için matplotlib kullanılmıştır.

## Algoritma Detayları

### Perceptron Öğrenme Kuralı:
```python
# Her yanlış tahmin için:
w = w + η * (y - ŷ) * x
b = b + η * (y - ŷ)
```

Burada:
- `η`: Learning rate
- `y`: Gerçek label (0 veya 1)
- `ŷ`: Tahmin edilen label
- `x`: Input vektörü