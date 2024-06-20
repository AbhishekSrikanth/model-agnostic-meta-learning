import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Task:

    def __init__(self, x, y, test_split: float = 0.2):
        self.test_split = test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_split)

    def get_train_data(self, K: int | None = None):

        if K is None:
            return self.x_train, self.y_train

        indices = np.random.choice(len(self.x_train), K, replace=False)
        x_train_k = self.x_train[indices]
        y_train_k = self.y_train[indices]
        return x_train_k, y_train_k

    def get_test_data(self, K: int | None = None):

        if K is None:
            return self.x_test, self.y_test

        indices = np.random.choice(len(self.x_test), K, replace=False)
        x_test_k = self.x_test[indices]
        y_test_k = self.y_test[indices]
        return x_test_k, y_test_k

    def __repr__(self) -> str:
        return '<BaseTask>'

    def plot_train_data(self):
        plt.scatter(self.x_train, self.y_train, label='Train Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    def plot_test_data(self):
        plt.scatter(self.x_test, self.y_test, label='Test Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()


class SinusoidTask(Task):

    def __init__(self, amplitude: float, phase: float, x, y, test_split: float = 0.2):
        self.amplitude = amplitude
        self.phase = phase
        self.plot_title = f'Sinusoid Task: Amplitude={
            self.amplitude}, Phase={self.phase}'
        super().__init__(x, y, test_split)

    def __repr__(self) -> str:
        return f'<SinusoidTask(amplitude={self.amplitude}, phase={self.phase})>'

    def plot_train_data(self):
        plt.scatter(self.x_train, self.y_train, label='Train Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(self.plot_title)
        plt.show()

    def plot_test_data(self):
        plt.scatter(self.x_test, self.y_test, label='Test Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(self.plot_title)
        plt.show()
