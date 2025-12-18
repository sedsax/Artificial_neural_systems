import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.widgets import Button, RadioButtons, TextBox, Slider  # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore

class LinearClassifier:
    def __init__(self):
        # Kendi random sayı üreticimiz (Linear Congruential Generator)
        self.seed = 12345
        
        # Model parametreleri
        self.weights = [self.random_small(), self.random_small()]  # w1, w2
        self.bias = self.random_small()  # b
        self.learning_rate = 0.01
        
        # Veri depolama
        self.points = {'class_0': [], 'class_1': []}
        self.current_class = 'class_0'
        
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
        
    def setup_ui(self):
        """Kullanıcı arayüzünü oluştur"""
        self.root = tk.Tk()
        self.root.title("Linear Classifier - Artificial Neural Systems")
        self.root.geometry("800x600")
        
        # Ana frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sol panel - Matplotlib grafiği
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-5, 5)  # -5 ile +5 arası (merkez 0,0)
        self.ax.set_ylim(-5, 5)  # -5 ile +5 arası (merkez 0,0)
        self.ax.set_xlabel('X Koordinatı')
        self.ax.set_ylabel('Y Koordinatı')
        self.ax.set_title('Linear Classification Demo')
        
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
        right_frame = tk.Frame(main_frame, width=200)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Class seçimi
        tk.Label(right_frame, text="Sınıf Seçimi:", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        self.class_var = tk.StringVar(value="Class 0")
        class_dropdown = ttk.Combobox(right_frame, textvariable=self.class_var, 
                                    values=["Class 0", "Class 1"], state="readonly")
        class_dropdown.pack(pady=(0, 15))
        class_dropdown.bind('<<ComboboxSelected>>', self.on_class_change)
        
        # Epoch ayarı
        tk.Label(right_frame, text="Epoch Sayısı:", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        self.epoch_var = tk.StringVar(value="100")
        epoch_entry = tk.Entry(right_frame, textvariable=self.epoch_var, width=10)
        epoch_entry.pack(pady=(0, 15))
        
        # Learning Rate ayarı
        tk.Label(right_frame, text="Learning Rate:", font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        self.lr_var = tk.StringVar(value="0.01")
        lr_entry = tk.Entry(right_frame, textvariable=self.lr_var, width=10)
        lr_entry.pack(pady=(0, 15))
        
        # Butonlar
        train_btn = tk.Button(right_frame, text="TRAIN", font=('Arial', 12, 'bold'),
                            bg='green', fg='white', command=self.train_model)
        train_btn.pack(pady=5, fill=tk.X)
        
        clear_btn = tk.Button(right_frame, text="CLEAR", font=('Arial', 12, 'bold'),
                            bg='red', fg='white', command=self.clear_data)
        clear_btn.pack(pady=5, fill=tk.X)
        
        # İstatistikler
        tk.Label(right_frame, text="İstatistikler:", font=('Arial', 12, 'bold')).pack(pady=(20, 5))
        self.stats_label = tk.Label(right_frame, text="", justify=tk.LEFT, 
                                  font=('Arial', 10))
        self.stats_label.pack(pady=(0, 10))
        
        self.update_stats()
        
    def on_class_change(self, event):
        """Sınıf değiştiğinde çağrılır"""
        if self.class_var.get() == "Class 0":
            self.current_class = 'class_0'
        else:
            self.current_class = 'class_1'
            
    def on_click(self, event):
        """Mouse click olayını işle"""
        if event.inaxes != self.ax:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Noktayı ekle
        self.points[self.current_class].append([x, y])
        
        # Grafiği güncelle
        self.update_plot()
        self.update_stats()
        
    def update_plot(self):
        """Grafiği güncelle"""
        self.ax.clear()
        self.ax.set_xlim(-5, 5)  # -5 ile +5 arası (merkez 0,0)
        self.ax.set_ylim(-5, 5)  # -5 ile +5 arası (merkez 0,0)
        self.ax.set_xlabel('X Koordinatı')
        self.ax.set_ylabel('Y Koordinatı')
        self.ax.set_title('Linear Classification Demo')
        
        # Grid ve koordinat eksenleri
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(y=0, color='black', linewidth=1.5)  # X ekseni
        self.ax.axvline(x=0, color='black', linewidth=1.5)  # Y ekseni
        
        # Origin noktası
        self.ax.plot(0, 0, 'ko', markersize=5)
        self.ax.text(0.1, 0.1, 'O(0,0)', fontsize=9)
        
        # Class 0 noktalarını çiz (mavi)
        if self.points['class_0']:
            x_coords = [point[0] for point in self.points['class_0']]
            y_coords = [point[1] for point in self.points['class_0']]
            self.ax.scatter(x_coords, y_coords, c='blue', s=50, label='Class 0', alpha=0.7)
        
        # Class 1 noktalarını çiz (kırmızı)
        if self.points['class_1']:
            x_coords = [point[0] for point in self.points['class_1']]
            y_coords = [point[1] for point in self.points['class_1']]
            self.ax.scatter(x_coords, y_coords, c='red', s=50, label='Class 1', alpha=0.7)
        
        # Decision boundary çiz (eğer eğitim yapıldıysa)
        if hasattr(self, 'trained') and self.trained:
            self.draw_decision_boundary()
            
        self.ax.legend()
        self.canvas.draw()
        
    def draw_decision_boundary(self):
        """Karar sınırını çiz"""
        if abs(self.weights[1]) < 1e-6:  # Dikey çizgi durumu
            if abs(self.weights[0]) > 1e-6:
                x_line = -self.bias / self.weights[0]
                if -5 <= x_line <= 5:  # Görünen alan içinde
                    self.ax.axvline(x=x_line, color='green', linewidth=2, 
                                  linestyle='--', label='Decision Boundary')
        else:
            # y = -(w1*x + b) / w2 formülü
            x_range = [i/10 for i in range(-50, 51)]  # -5'den +5'e 0.1'lik adımlar
            valid_x = []
            valid_y = []
            
            for x in x_range:
                y = -(self.weights[0] * x + self.bias) / self.weights[1]
                if -5 <= y <= 5:  # Görünen alan içinde (-5,+5)
                    valid_x.append(x)
                    valid_y.append(y)
            
            if valid_x:
                self.ax.plot(valid_x, valid_y, 'g--', linewidth=2, label='Decision Boundary')
    
    def train_model(self):
        """Modeli eğit"""
        # Veri kontrolü
        if len(self.points['class_0']) == 0 or len(self.points['class_1']) == 0:
            messagebox.showwarning("Uyarı", "Her iki sınıftan da en az bir nokta eklemelisiniz!")
            return
            
        # Parametreleri al
        try:
            epochs = int(self.epoch_var.get())
            self.learning_rate = float(self.lr_var.get())
        except ValueError:
            messagebox.showerror("Hata", "Geçersiz epoch veya learning rate değeri!")
            return
        
        # Veriyi hazırla
        X, y = self.prepare_training_data()
        
        # Perceptron algoritması
        print(f"Eğitim başlıyor... (Maksimum {epochs} epoch)")
        print(f"Veri: {len(X)} nokta, Learning Rate: {self.learning_rate}")
        
        # Animasyon için decision boundary'yi her epoch'ta güncelle
        self.trained = True  # Grafik güncellemesi için gerekli
        
        for epoch in range(epochs):
            errors = 0
            old_weights = [self.weights[0], self.weights[1]]
            old_bias = self.bias
            
            for i in range(len(X)):
                # Tahmin yap
                prediction = self.predict_single(X[i])
                
                # Hata varsa ağırlıkları güncelle
                if prediction != y[i]:
                    errors += 1
                    # w = w + η * (y - ŷ) * x
                    error = y[i] - prediction
                    self.weights[0] += self.learning_rate * error * X[i][0]
                    self.weights[1] += self.learning_rate * error * X[i][1]
                    self.bias += self.learning_rate * error
            
            # Ağırlıklar değiştiyse grafiği güncelle (canlı animasyon)
            weights_changed = (old_weights[0] != self.weights[0] or 
                             old_weights[1] != self.weights[1] or 
                             old_bias != self.bias)
            
            if weights_changed or epoch == 0:
                self.update_plot()
                self.ax.set_title(f'Linear Classification - Epoch {epoch} (Hatalar: {errors})')
                self.canvas.draw()
                
                # Animasyon hızı için kısa bekle (saf Python ile)
                self.root.update()
                self.simple_sleep(0.1)  # 100ms bekle
            
            # Her epochtaki durumu göster
            accuracy = (len(X) - errors) / len(X) * 100
            if epoch < 10 or epoch % 5 == 0 or errors == 0:
                print(f"Epoch {epoch}: Hatalar = {errors}, Accuracy = {accuracy:.1f}%")
                
            # Eğer hata yoksa dur
            if errors == 0:
                print(f"✅ Mükemmel! Tüm noktalar doğru sınıflandırıldı.")
                print(f"🎯 Eğitim tamamlandı! Toplam epoch: {epoch + 1}")
                self.ax.set_title(f'Linear Classification - TAMAMLANDI! (Epoch: {epoch + 1})')
                self.canvas.draw()
                break
        else:
            # For döngüsü break ile çıkmadıysa (maksimum epocha ulaşıldıysa)
            print(f"⚠️ Maksimum epoch ({epochs}) sayısına ulaşıldı.")
            final_accuracy = (len(X) - errors) / len(X) * 100
            print(f"📊 Final accuracy: {final_accuracy:.1f}%")
            self.ax.set_title(f'Linear Classification - Max Epoch Ulaşıldı (Acc: {final_accuracy:.1f}%)')
            self.canvas.draw()
        
        self.trained = True
        self.update_plot()
        self.update_stats()
        
    def prepare_training_data(self):
        """Eğitim verisini hazırla"""
        X = []
        y = []
        
        # Class 0 noktaları (label = 0)
        for point in self.points['class_0']:
            X.append(point)
            y.append(0)
            
        # Class 1 noktaları (label = 1)  
        for point in self.points['class_1']:
            X.append(point)
            y.append(1)
            
        return X, y
    
    def predict_single(self, x):
        """Tek bir nokta için tahmin yap"""
        # z = w1*x1 + w2*x2 + b
        z = self.weights[0] * x[0] + self.weights[1] * x[1] + self.bias
        return 1 if z >= 0 else 0
    
    def clear_data(self):
        """Tüm veriyi temizle"""
        self.points = {'class_0': [], 'class_1': []}
        self.weights = [self.random_small(), self.random_small()]
        self.bias = self.random_small()
        self.trained = False
        self.update_plot()
        self.update_stats()
        
    def update_stats(self):
        """İstatistikleri güncelle"""
        class_0_count = len(self.points['class_0'])
        class_1_count = len(self.points['class_1'])
        total = class_0_count + class_1_count
        
        stats_text = f"Class 0: {class_0_count} nokta\n"
        stats_text += f"Class 1: {class_1_count} nokta\n"
        stats_text += f"Toplam: {total} nokta\n\n"
        
        if hasattr(self, 'trained') and self.trained:
            stats_text += f"Model Parametreleri:\n"
            stats_text += f"w1 = {self.weights[0]:.3f}\n"
            stats_text += f"w2 = {self.weights[1]:.3f}\n"
            stats_text += f"b = {self.bias:.3f}\n"
        
        self.stats_label.config(text=stats_text)
    
    def run(self):
        """Uygulamayı çalıştır"""
        self.root.mainloop()

if __name__ == "__main__":
    app = LinearClassifier()
    app.run()