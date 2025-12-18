import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.widgets import Button, RadioButtons, TextBox, Slider  # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore

class MultiClassClassifier:
    def __init__(self):
        # Kendi random sayı üreticimiz (Linear Congruential Generator)
        self.seed = 12345
        
        # Model parametreleri - Her sınıf için ayrı perceptron
        self.num_classes = 10
        self.weights = []  # Her sınıf için [w1, w2] ağırlıkları
        self.biases = []   # Her sınıf için bias
        
        # Her sınıf için ayrı perceptron oluştur
        for i in range(self.num_classes):
            self.weights.append([self.random_small(), self.random_small()])
            self.biases.append(self.random_small())
            
        self.learning_rate = 0.01
        
        # Veri depolama - Her sınıf için ayrı liste
        self.points = {}
        for i in range(self.num_classes):
            self.points[f'class_{i}'] = []
            
        self.current_class = 0
        
        # Normalizasyon parametreleri
        self.normalize_enabled = True
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None
        
        # Renk paleti - 10 farklı renk
        self.colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
                      '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A0522D']
        
        # UI bileşenleri
        self.setup_ui()
    
    def random_number(self):
        """Basit Linear Congruential Generator"""
        self.seed = (self.seed * 1103515245 + 12345) % (2**31)
        return self.seed / (2**31)
    
    def random_small(self):
        """Küçük random sayı üret (-0.01, 0.01 arası)"""
        return (self.random_number() - 0.5) * 0.02
    
    def simple_sleep(self, seconds):
        """Saf Python ile sleep (time kütüphanesi kullanmadan)"""
        start_time = self.root.tk.call('clock', 'milliseconds')  
        target_time = start_time + (seconds * 1000)
        while self.root.tk.call('clock', 'milliseconds') < target_time:
            self.root.update_idletasks()
    
    def calculate_normalization_params(self, all_points):
        """Normalizasyon parametrelerini hesapla"""
        if not all_points:
            return
            
        x_coords = [point[0] for point in all_points]
        y_coords = [point[1] for point in all_points]
        
        self.x_min = min(x_coords)
        self.x_max = max(x_coords)
        self.y_min = min(y_coords)
        self.y_max = max(y_coords)
        
        print(f"📏 Normalizasyon Parametreleri:")
        print(f"  X aralığı: [{self.x_min:.2f}, {self.x_max:.2f}]")
        print(f"  Y aralığı: [{self.y_min:.2f}, {self.y_max:.2f}]")
    
    def normalize_point(self, point):
        """Tek bir noktayı normalize et (Min-Max Scaling: 0-1 arası)"""
        if not self.normalize_enabled:
            return point
            
        x, y = point[0], point[1]
        
        # Min-Max normalization: (x - min) / (max - min)
        if self.x_max != self.x_min:
            x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        else:
            x_norm = 0.5  # Tüm X değerleri aynıysa ortaya koy
            
        if self.y_max != self.y_min:
            y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        else:
            y_norm = 0.5  # Tüm Y değerleri aynıysa ortaya koy
            
        return [x_norm, y_norm]
    
    def denormalize_point(self, normalized_point):
        """Normalize edilmiş noktayı orijinal koordinatlara çevir"""
        if not self.normalize_enabled:
            return normalized_point
            
        x_norm, y_norm = normalized_point[0], normalized_point[1]
        
        # Ters Min-Max: x = normalized * (max - min) + min
        x_orig = x_norm * (self.x_max - self.x_min) + self.x_min
        y_orig = y_norm * (self.y_max - self.y_min) + self.y_min
        
        return [x_orig, y_orig]
    
    def normalize_dataset(self, points):
        """Tüm dataset'i normalize et"""
        if not self.normalize_enabled:
            return points
            
        normalized_points = []
        for point in points:
            normalized_points.append(self.normalize_point(point))
        return normalized_points
        
    def setup_ui(self):
        """Kullanıcı arayüzünü oluştur"""
        self.root = tk.Tk()
        self.root.title("Multi-Class Classifier (10 Classes) - Artificial Neural Systems")
        self.root.geometry("1000x700")
        
        # Ana frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sol panel - Matplotlib grafiği
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-5, 5)  # -5 ile +5 arası (merkez 0,0)
        self.ax.set_ylim(-5, 5)  # -5 ile +5 arası (merkez 0,0)
        self.ax.set_xlabel('X Koordinatı')
        self.ax.set_ylabel('Y Koordinatı')
        self.ax.set_title('Multi-Class Classification Demo (10 Classes)')
        
        # Grid ve koordinat eksenleri
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(y=0, color='black', linewidth=1.5)  # X ekseni
        self.ax.axvline(x=0, color='black', linewidth=1.5)  # Y ekseni
        
        # Origin noktası
        self.ax.plot(0, 0, 'ko', markersize=5)
        self.ax.text(0.1, 0.1, 'O(0,0)', fontsize=9)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mouse click event
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Sağ panel - Kontroller
        right_frame = tk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Class seçimi
        tk.Label(right_frame, text="Sınıf Seçimi:", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        
        # Dropdown için class listesi
        class_options = [f"Class {i}" for i in range(self.num_classes)]
        self.class_var = tk.StringVar(value="Class 0")
        class_dropdown = ttk.Combobox(right_frame, textvariable=self.class_var, 
                                    values=class_options, state="readonly", width=15)
        class_dropdown.pack(pady=(0, 10))
        class_dropdown.bind('<<ComboboxSelected>>', self.on_class_change)
        
        # Renk göstergesi
        self.color_label = tk.Label(right_frame, text="●", font=('Arial', 20), 
                                   fg=self.colors[0])
        self.color_label.pack(pady=(0, 15))
        
        # Epoch ayarı
        tk.Label(right_frame, text="Epoch Sayısı:", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        self.epoch_var = tk.StringVar(value="50")
        epoch_entry = tk.Entry(right_frame, textvariable=self.epoch_var, width=10)
        epoch_entry.pack(pady=(0, 15))
        
        # Learning Rate ayarı
        tk.Label(right_frame, text="Learning Rate:", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        self.lr_var = tk.StringVar(value="0.01")
        lr_entry = tk.Entry(right_frame, textvariable=self.lr_var, width=10)
        lr_entry.pack(pady=(0, 15))
        
        # Normalizasyon seçeneği
        self.normalize_var = tk.BooleanVar(value=True)
        normalize_check = tk.Checkbutton(right_frame, text="Veriyi Normalize Et", 
                                       variable=self.normalize_var, font=('Arial', 10),
                                       command=self.toggle_normalization)
        normalize_check.pack(pady=(0, 15))
        
        # Butonlar
        train_btn = tk.Button(right_frame, text="TRAIN", font=('Arial', 12, 'bold'),
                            bg='green', fg='white', command=self.train_model)
        train_btn.pack(pady=5, fill=tk.X)
        
        clear_btn = tk.Button(right_frame, text="CLEAR", font=('Arial', 12, 'bold'),
                            bg='red', fg='white', command=self.clear_data)
        clear_btn.pack(pady=5, fill=tk.X)
        
        # Test butonu
        test_btn = tk.Button(right_frame, text="TEST POINT", font=('Arial', 11, 'bold'),
                           bg='orange', fg='white', command=self.toggle_test_mode)
        test_btn.pack(pady=5, fill=tk.X)
        
        # Test modu göstergesi
        self.test_mode = False
        self.test_label = tk.Label(right_frame, text="Normal Mode", 
                                 font=('Arial', 10), fg='blue')
        self.test_label.pack(pady=(5, 15))
        
        # İstatistikler
        tk.Label(right_frame, text="İstatistikler:", font=('Arial', 12, 'bold')).pack(pady=(20, 5))
        self.stats_label = tk.Label(right_frame, text="", justify=tk.LEFT, 
                                  font=('Arial', 9), wraplength=230)
        self.stats_label.pack(pady=(0, 10))
        
        # Sınıf renkleri göstergesi
        tk.Label(right_frame, text="Sınıf Renkleri:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        colors_frame = tk.Frame(right_frame)
        colors_frame.pack(pady=(0, 10))
        
        for i in range(self.num_classes):
            row = i // 5  # Her satırda 5 sınıf
            col = i % 5
            
            class_frame = tk.Frame(colors_frame)
            if col == 0:  # Yeni satır
                class_frame.pack(anchor='w', pady=2)
            else:
                class_frame.pack(side=tk.LEFT, padx=5)
                
            color_dot = tk.Label(class_frame, text="●", font=('Arial', 12), 
                               fg=self.colors[i])
            color_dot.pack(side=tk.LEFT)
            
            class_text = tk.Label(class_frame, text=f"C{i}", font=('Arial', 8))
            class_text.pack(side=tk.LEFT)
            
            if col == 4:  # Satır sonu
                colors_frame = tk.Frame(right_frame)
                colors_frame.pack(pady=(0, 10))
        
        self.update_stats()
        
    def on_class_change(self, event):
        """Sınıf değiştiğinde çağrılır"""
        class_text = self.class_var.get()
        self.current_class = int(class_text.split()[1])
        self.color_label.config(fg=self.colors[self.current_class])
        
    def toggle_normalization(self):
        """Normalizasyon modunu aç/kapat"""
        self.normalize_enabled = self.normalize_var.get()
        status = "AÇIK" if self.normalize_enabled else "KAPALI"
        print(f"📊 Veri Normalizasyonu: {status}")
        
    def toggle_test_mode(self):
        """Test modunu aç/kapat"""
        self.test_mode = not self.test_mode
        if self.test_mode:
            self.test_label.config(text="TEST MODE", fg='red')
        else:
            self.test_label.config(text="Normal Mode", fg='blue')
            
    def on_click(self, event):
        """Mouse click olayını işle"""
        if event.inaxes != self.ax:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        if self.test_mode:
            # Test modu - Bu noktanın hangi sınıfa ait olduğunu tahmin et
            if hasattr(self, 'trained') and self.trained:
                # Orijinal koordinatlarda tıklanan noktayı test et
                original_point = [x, y]
                
                # Eğer normalize görselleştirme modundaysak, koordinatları dönüştür
                if (self.normalize_enabled and hasattr(self, 'x_min')):
                    # Tıklanan nokta normalize edilmiş koordinat sisteminde
                    # Onu orijinal koordinatlara çevir
                    original_point = self.denormalize_point([x, y])
                
                predicted_class = self.predict_point(original_point)
                self.ax.plot(x, y, marker='*', markersize=15, 
                           color=self.colors[predicted_class], 
                           markeredgecolor='black', markeredgewidth=2)
                
                # Text offset'i koordinat sistemine göre ayarla
                if (self.normalize_enabled and hasattr(self, 'x_min')):
                    text_offset = 0.05  # Normalize sistemde küçük offset
                else:
                    text_offset = 0.2   # Orijinal sistemde büyük offset
                    
                self.ax.text(x + text_offset, y + text_offset, f'C{predicted_class}', 
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                self.canvas.draw()
                
                if (self.normalize_enabled and hasattr(self, 'x_min')):
                    print(f"Test noktası Norm:({x:.3f}, {y:.3f}) Orig:({original_point[0]:.2f}, {original_point[1]:.2f}) -> Class {predicted_class}")
                else:
                    print(f"Test noktası ({x:.2f}, {y:.2f}) -> Tahmin: Class {predicted_class}")
            else:
                messagebox.showwarning("Uyarı", "Önce modeli eğitmelisiniz!")
        else:
            # Normal mod - Nokta ekle (her zaman orijinal koordinatlarda sakla)
            self.points[f'class_{self.current_class}'].append([x, y])
            self.update_plot()
            self.update_stats()
        
    def update_plot(self):
        """Grafiği güncelle"""
        self.ax.clear()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_xlabel('X Koordinatı')
        self.ax.set_ylabel('Y Koordinatı')
        self.ax.set_title('Multi-Class Classification Demo (10 Classes)')
        
        # Grid ve koordinat eksenleri
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(y=0, color='black', linewidth=1.5)
        self.ax.axvline(x=0, color='black', linewidth=1.5)
        
        # Origin noktası - koordinat sistemine göre ayarla
        if (self.normalize_enabled and hasattr(self, 'trained') and 
            self.trained and hasattr(self, 'x_min')):
            # Normalize edilmiş sistemde (0,0) ve (1,1) köşeleri işaretle
            self.ax.plot(0, 0, 'ko', markersize=5)
            self.ax.plot(1, 1, 'ko', markersize=3, alpha=0.5)
            self.ax.text(0.02, 0.02, 'Min', fontsize=8)
            self.ax.text(0.98, 0.98, 'Max', fontsize=8)
        else:
            # Orijinal sistemde merkez (0,0)
            self.ax.plot(0, 0, 'ko', markersize=5)
            self.ax.text(0.1, 0.1, 'O(0,0)', fontsize=9)
        
        # Her sınıfın noktalarını çiz
        legend_elements = []
        for i in range(self.num_classes):
            if self.points[f'class_{i}']:
                # Eğer normalizasyon açık ve eğitim yapıldıysa normalize edilmiş koordinatları göster
                if (self.normalize_enabled and hasattr(self, 'trained') and 
                    self.trained and hasattr(self, 'x_min')):
                    # Noktaları normalize et ve 0-1 aralığında göster
                    x_coords = []
                    y_coords = []
                    for point in self.points[f'class_{i}']:
                        norm_point = self.normalize_point(point)
                        x_coords.append(norm_point[0])
                        y_coords.append(norm_point[1])
                    
                    # Eksenleri 0-1 aralığına ayarla
                    self.ax.set_xlim(-0.1, 1.1)
                    self.ax.set_ylim(-0.1, 1.1)
                    self.ax.set_title('Multi-Class Classification (Normalize Edilmiş)')
                else:
                    # Orijinal koordinatları göster
                    x_coords = [point[0] for point in self.points[f'class_{i}']]
                    y_coords = [point[1] for point in self.points[f'class_{i}']]
                
                scatter = self.ax.scatter(x_coords, y_coords, c=self.colors[i], 
                                        s=60, label=f'Class {i}', alpha=0.8,
                                        edgecolors='black', linewidth=1)
                legend_elements.append(scatter)
        
        # Decision boundaries çiz (eğer eğitim yapıldıysa)
        if hasattr(self, 'trained') and self.trained:
            self.draw_decision_boundaries()
            
        # Legend sadece veri olan sınıflar için
        if legend_elements:
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        self.canvas.draw()
        
    def draw_decision_boundaries(self):
        """Tüm sınıflar için karar sınırlarını çiz"""
        # Her sınıf için decision boundary çiz
        for class_idx in range(self.num_classes):
            # Bu sınıfın noktası varsa boundary çiz
            if self.points[f'class_{class_idx}']:
                self.draw_single_boundary(class_idx)
    
    def draw_single_boundary(self, class_idx):
        """Tek bir sınıf için karar sınırını çiz"""
        w1, w2 = self.weights[class_idx]
        b = self.biases[class_idx]
        
        # Koordinat sistemi sınırlarını belirle
        if (self.normalize_enabled and hasattr(self, 'trained') and 
            self.trained and hasattr(self, 'x_min')):
            # Normalize edilmiş koordinat sistemi (0-1)
            x_min, x_max = -0.1, 1.1
            y_min, y_max = -0.1, 1.1
            x_range = [i/100 for i in range(-10, 111)]  # 0.01'lik adımlar
        else:
            # Orijinal koordinat sistemi (-5, +5)
            x_min, x_max = -5, 5
            y_min, y_max = -5, 5
            x_range = [i/10 for i in range(-50, 51)]  # 0.1'lik adımlar
        
        if abs(w2) < 1e-6:  # Dikey çizgi durumu
            if abs(w1) > 1e-6:
                x_line = -b / w1
                if x_min <= x_line <= x_max:
                    self.ax.axvline(x=x_line, color=self.colors[class_idx], 
                                  linewidth=1.5, linestyle='--', alpha=0.7)
        else:
            # y = -(w1*x + b) / w2 formülü
            valid_x = []
            valid_y = []
            
            for x in x_range:
                y = -(w1 * x + b) / w2
                if y_min <= y <= y_max:
                    valid_x.append(x)
                    valid_y.append(y)
            
            if valid_x:
                self.ax.plot(valid_x, valid_y, color=self.colors[class_idx], 
                           linewidth=1.5, linestyle='--', alpha=0.7)
    
    def train_model(self):
        """Modeli eğit - One-vs-All stratejisi"""
        # Veri kontrolü
        total_points = sum(len(self.points[f'class_{i}']) for i in range(self.num_classes))
        if total_points < 2:
            messagebox.showwarning("Uyarı", "En az 2 sınıftan nokta eklemelisiniz!")
            return
            
        # Parametreleri al
        try:
            epochs = int(self.epoch_var.get())
            self.learning_rate = float(self.lr_var.get())
        except ValueError:
            messagebox.showerror("Hata", "Geçersiz epoch veya learning rate değeri!")
            return
        
        print(f"🎯 Multi-Class Eğitim Başlıyor...")
        print(f"📊 Toplam nokta: {total_points}, Learning Rate: {self.learning_rate}")
        print(f"🔢 Strateji: One-vs-All (Her sınıf için ayrı perceptron)")
        print(f"📏 Normalizasyon: {'AÇIK' if self.normalize_enabled else 'KAPALI'}")
        
        # Tüm veriyi hazırla
        all_points, all_labels = self.prepare_all_data()
        
        # Normalizasyon parametrelerini hesapla
        if self.normalize_enabled:
            self.calculate_normalization_params(all_points)
            # Veriyi normalize et
            all_points = self.normalize_dataset(all_points)
            print(f"✅ Veri normalize edildi (0-1 aralığına)")
        
        # Her sınıf için ayrı ayrı eğit (One-vs-All)
        self.trained = True
        
        for class_idx in range(self.num_classes):
            # Bu sınıf için binary labels oluştur
            binary_labels = [1 if label == class_idx else 0 for label in all_labels]
            
            # Bu sınıfın verisi varsa eğit
            if 1 in binary_labels:
                print(f"\n🔄 Class {class_idx} eğitiliyor...")
                self.train_single_perceptron(class_idx, all_points, binary_labels, epochs)
        
        # Final accuracy hesapla
        self.calculate_final_accuracy(all_points, all_labels)
        self.update_plot()
        print(f"✅ Tüm sınıflar eğitildi!")
        
    def train_single_perceptron(self, class_idx, X, binary_labels, epochs):
        """Tek bir perceptron eğit (One-vs-All için)"""
        for epoch in range(epochs):
            errors = 0
            
            for i in range(len(X)):
                # Bu sınıf için tahmin yap
                prediction = self.predict_single_class(X[i], class_idx)
                
                # Hata varsa ağırlıkları güncelle
                if prediction != binary_labels[i]:
                    errors += 1
                    error = binary_labels[i] - prediction
                    self.weights[class_idx][0] += self.learning_rate * error * X[i][0]
                    self.weights[class_idx][1] += self.learning_rate * error * X[i][1]
                    self.biases[class_idx] += self.learning_rate * error
            
            # İlk 10 epoch veya her 10 epochta bir yazdır
            if epoch < 10 or epoch % 10 == 0 or errors == 0:
                accuracy = (len(X) - errors) / len(X) * 100
                print(f"  Epoch {epoch}: Hatalar = {errors}, Binary Acc = {accuracy:.1f}%")
                
            # Animasyon için grafik güncelle
            if epoch % 5 == 0:
                self.update_plot()
                self.ax.set_title(f'Training Class {class_idx} - Epoch {epoch}')
                self.canvas.draw()
                self.root.update()
                self.simple_sleep(0.05)
                
            # Hata yoksa bu sınıf için dur
            if errors == 0:
                print(f"  ✅ Class {class_idx} mükemmel öğrenildi! (Epoch: {epoch})")
                break
    
    def predict_single_class(self, x, class_idx):
        """Tek bir sınıf için tahmin yap (binary)"""
        z = self.weights[class_idx][0] * x[0] + self.weights[class_idx][1] * x[1] + self.biases[class_idx]
        return 1 if z >= 0 else 0
    
    def predict_point(self, x):
        """Bir nokta için en yüksek skorlu sınıfı döndür"""
        # Test noktasını normalize et (eğer normalizasyon açıksa)
        if self.normalize_enabled and hasattr(self, 'x_min'):
            x_normalized = self.normalize_point(x)
        else:
            x_normalized = x
            
        scores = []
        for class_idx in range(self.num_classes):
            score = (self.weights[class_idx][0] * x_normalized[0] + 
                    self.weights[class_idx][1] * x_normalized[1] + 
                    self.biases[class_idx])
            scores.append(score)
        
        # En yüksek skora sahip sınıfı döndür
        return scores.index(max(scores))
    
    def prepare_all_data(self):
        """Tüm veriyi hazırla"""
        X = []
        y = []
        
        for class_idx in range(self.num_classes):
            for point in self.points[f'class_{class_idx}']:
                X.append(point)
                y.append(class_idx)
                
        return X, y
    
    def calculate_final_accuracy(self, X, y):
        """Final accuracy hesapla"""
        if not X:
            return
            
        correct = 0
        for i in range(len(X)):
            predicted = self.predict_point(X[i])
            if predicted == y[i]:
                correct += 1
        
        accuracy = correct / len(X) * 100
        print(f"\n📊 FINAL MULTI-CLASS ACCURACY: {accuracy:.1f}% ({correct}/{len(X)})")
        
        # Sınıf başına accuracy
        for class_idx in range(self.num_classes):
            class_points = [i for i, label in enumerate(y) if label == class_idx]
            if class_points:
                class_correct = sum(1 for i in class_points if self.predict_point(X[i]) == class_idx)
                class_acc = class_correct / len(class_points) * 100
                print(f"  Class {class_idx}: {class_acc:.1f}% ({class_correct}/{len(class_points)})")
    
    def clear_data(self):
        """Tüm veriyi temizle"""
        for i in range(self.num_classes):
            self.points[f'class_{i}'] = []
            self.weights[i] = [self.random_small(), self.random_small()]
            self.biases[i] = self.random_small()
            
        self.trained = False
        self.test_mode = False
        self.test_label.config(text="Normal Mode", fg='blue')
        self.update_plot()
        self.update_stats()
        
    def update_stats(self):
        """İstatistikleri güncelle"""
        stats_text = "Sınıf Nokta Sayıları:\n"
        total = 0
        
        for i in range(self.num_classes):
            count = len(self.points[f'class_{i}'])
            if count > 0:
                stats_text += f"Class {i}: {count} nokta\n"
                total += count
        
        stats_text += f"\nToplam: {total} nokta\n"
        
        # Normalizasyon durumu
        norm_status = "AÇIK" if self.normalize_enabled else "KAPALI"
        stats_text += f"📏 Normalize: {norm_status}\n"
        
        if hasattr(self, 'trained') and self.trained:
            stats_text += f"\n🤖 Model eğitildi!\n"
            if hasattr(self, 'x_min') and self.normalize_enabled:
                stats_text += f"📊 Veri aralığı:\n"
                stats_text += f"  X: [{self.x_min:.2f}, {self.x_max:.2f}]\n"
                stats_text += f"  Y: [{self.y_min:.2f}, {self.y_max:.2f}]\n"
            stats_text += f"⭐ Test modu aktif"
        
        self.stats_label.config(text=stats_text)
    
    def run(self):
        """Uygulamayı çalıştır"""
        self.root.mainloop()

if __name__ == "__main__":
    app = MultiClassClassifier()
    app.run()