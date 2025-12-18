import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons

class ModelTrainer:
    def __init__(self):
        self.data = []  # Stores (x, y, class_label)
        self.current_class = 0
        self.centroids = {}
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Click to mark data points. Train and Test using buttons.")
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def set_class(self, class_label):
        self.current_class = class_label
        print(f"Current class set to: {class_label}")

    def onclick(self, event):
        if event.xdata is not None and event.ydata is not None:  # Ensure click is within plot bounds
            self.data.append((event.xdata, event.ydata, self.current_class))
            print(f"Point added: ({event.xdata:.2f}, {event.ydata:.2f}) with class {self.current_class}")
            self.ax.scatter(event.xdata, event.ydata, c=f"C{self.current_class}", label=f"Class {self.current_class}")
            self.ax.legend()
            self.fig.canvas.draw()

    def train(self, event):
        if not self.data:
            print("No data to train on.")
            return

        # Prepare data
        data = np.array(self.data)
        X = data[:, :2]  # Coordinates
        y = data[:, 2]   # Class labels

        # Calculate centroids for each class
        self.centroids = {}
        for cls in np.unique(y):
            class_points = X[y == cls]
            self.centroids[cls] = np.mean(class_points, axis=0)

        print("Training complete. Centroids:", self.centroids)

        # Plot decision boundaries (for visualization purposes only)
        for cls, centroid in self.centroids.items():
            self.ax.scatter(*centroid, c=f"C{int(cls)}", marker='x', s=100, label=f"Centroid Class {int(cls)}")
        self.ax.legend()
        self.ax.set_title("Training Complete with Centroids")
        self.fig.canvas.draw()

    def test(self, event):
        if not self.centroids:
            print("Model not trained yet.")
            return

        # Example test data
        test_data = np.array([[2, 2], [7, 7]])
        predictions = []
        for point in test_data:
            distances = {cls: np.linalg.norm(point - centroid) for cls, centroid in self.centroids.items()}
            predicted_class = min(distances, key=distances.get)
            predictions.append(predicted_class)
            self.ax.scatter(*point, c=f"C{int(predicted_class)}", marker='o', edgecolors='k', s=100, label=f"Test Point Class {int(predicted_class)}")
        print("Predictions for test data:", predictions)
        self.ax.legend()
        self.fig.canvas.draw()

    def create_interface(self):
        # Add class selection radio buttons
        rax = plt.axes([0.01, 0.4, 0.1, 0.4])
        radio = RadioButtons(rax, [str(i) for i in range(10)])

        def class_callback(label):
            self.set_class(int(label))

        radio.on_clicked(class_callback)

        # Add train button
        train_ax = plt.axes([0.8, 0.05, 0.1, 0.075])
        train_button = Button(train_ax, 'Train')
        train_button.on_clicked(self.train)

        # Add test button
        test_ax = plt.axes([0.65, 0.05, 0.1, 0.075])
        test_button = Button(test_ax, 'Test')
        test_button.on_clicked(self.test)

        plt.show()

# Example usage
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.create_interface()