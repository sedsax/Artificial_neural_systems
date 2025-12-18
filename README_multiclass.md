# Multi-Class Classifier - 10 Sınıf Desteği

Bu proje, **One-vs-All** stratejisi kullanarak 10 farklı sınıfı ayırabilen çoklu sınıf sınıflandırıcısının görsel implementasyonudur.

## 🎯 Özellikler

### 🎨 Çoklu Sınıf Desteği
- **10 farklı sınıf**: Class 0'dan Class 9'a kadar
- **Renkli görselleştirme**: Her sınıf için farklı renk
- **Dropdown seçimi**: Hangi sınıfı işaretlemek istediğinizi seçin

### 🧠 One-vs-All Algoritması
- **10 ayrı perceptron**: Her sınıf için ayrı ikili sınıflandırıcı
- **Saf Python implementasyonu**: Hiçbir ML kütüphanesi kullanılmadı
- **Canlı eğitim animasyonu**: Decision boundary'lerin gelişimini izleyin

### 🎮 İnteraktif Özellikler
- **Mouse ile veri girişi**: Koordinat sisteminde noktaları işaretleyin
- **Test modu**: Eğitim sonrası yeni noktaları test edin
- **Real-time görselleştirme**: Eğitim sırasında canlı güncelleme

## 🚀 Nasıl Kullanılır?

### 1. Veri Ekleme
```
1. Dropdown'dan bir sınıf seçin (Class 0-9)
2. Mouse ile koordinat sisteminde noktaları işaretleyin
3. Farklı sınıflar seçerek farklı bölgelere noktalar ekleyin
```

### 2. Model Eğitimi
```
1. TRAIN butonuna basın
2. Her sınıf için ayrı ayrı eğitim başlar
3. Decision boundary'lerin oluşmasını izleyin
```

### 3. Test Etme
```
1. TEST POINT butonuna basın (Test moduna geçin)
2. Mouse ile herhangi bir yere tıklayın
3. Modelin tahminini görün (yıldız işareti + sınıf etiketi)
```

## 🎨 Renk Paleti

| Sınıf | Renk | Hex Code |
|-------|------|----------|
| Class 0 | 🔴 Kırmızı | #FF0000 |
| Class 1 | 🟢 Yeşil | #00FF00 |
| Class 2 | 🔵 Mavi | #0000FF |
| Class 3 | 🟡 Sarı | #FFFF00 |
| Class 4 | 🟣 Magenta | #FF00FF |
| Class 5 | 🔷 Cyan | #00FFFF |
| Class 6 | 🟠 Turuncu | #FFA500 |
| Class 7 | 🟣 Mor | #800080 |
| Class 8 | 🌸 Pembe | #FFC0CB |
| Class 9 | 🟤 Kahverengi | #A0522D |

## 🧮 Algoritma Detayları

### One-vs-All Stratejisi
Her sınıf için ikili sınıflandırıcı:
- **Class i vs Diğerleri**: Sınıf i → 1, Diğer tüm sınıflar → 0
- **10 ayrı perceptron**: Her biri w₁x + w₂y + b = 0 öğrenir
- **Tahmin**: En yüksek skora sahip sınıf seçilir

### Matematiksel Formül
```python
# Her sınıf i için:
score_i = w₁ᵢ * x + w₂ᵢ * y + bᵢ

# Final tahmin:
predicted_class = argmax(score_i)
```

### Eğitim Süreci
```python
# Her sınıf için ayrı ayrı:
for each_class in range(10):
    # Binary labels oluştur
    labels = [1 if actual_class == each_class else 0]
    
    # Perceptron eğit
    for epoch in range(max_epochs):
        # Ağırlık güncellemeleri...
```

## 📊 Performans Metrikleri

### Genel Accuracy
- **Multi-class accuracy**: Tüm sınıflar için genel başarı oranı
- **Sınıf bazında accuracy**: Her sınıfın kendi başarı oranı

### Görsel Feedback
- **Epoch takibi**: Hangi sınıfın hangi epochta eğitildiği
- **Hata sayısı**: Her epochtaki yanlış tahmin sayısı
- **Decision boundaries**: Her sınıf için farklı renkte kesikli çizgiler

## 🎓 Eğitim Amaçları

Bu proje şunları öğretir:
- **Çoklu sınıf sınıflandırma** stratejileri
- **One-vs-All** yaklaşımının implementasyonu
- **Perceptron algoritmasının** genişletilmesi
- **Görsel makine öğrenmesi** uygulamaları

## 🔧 Teknik Özellikler

- **Saf Python**: Sadece matplotlib ve tkinter
- **Kendi random generator**: Linear Congruential Generator
- **Memory efficient**: Her sınıf için ayrı veri yapıları
- **Real-time animation**: Canlı eğitim görselleştirmesi

## 🎯 Kullanım Senaryoları

### Basit Test
1. Her sınıftan 2-3 nokta ekleyin
2. Noktaları farklı bölgelere yerleştirin
3. Kısa epoch (20-30) ile eğitin

### Karmaşık Test
1. 5-6 sınıftan çok sayıda nokta ekleyin
2. Sınıfları iç içe geçirecek şekilde yerleştirin
3. Yüksek epoch (100+) ile eğitin

### Test Modu
1. Eğitim tamamlandıktan sonra TEST POINT aktif edin
2. Farklı bölgelere tıklayarak tahminleri görün
3. Decision boundary'lerin doğruluğunu test edin

Bu şekilde gerçek bir çoklu sınıf makine öğrenmesi sistemini sıfırdan öğrenmiş olursunuz!