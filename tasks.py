import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Task:

    def __init__(self, x, y, test_split: float = 0.2):
        self.test_split = test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_split)

    def _get_data(self, x: list, y: list, K: int | None = None):
        if K is None:
            return x, y

        indices = np.random.choice(len(x), K, replace=False)
        x_k = x[indices]
        y_k = y[indices]
        return x_k, y_k

    def _plot_data(self, x, y, label):
        plt.scatter(x, y, label=label)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    def get_train_data(self, K: int | None = None):
        return self._get_data(self.x_train, self.y_train, K)

    def get_test_data(self, K: int | None = None):
        return self._get_data(self.x_test, self.y_test, K)

    def plot_train_data(self):
        self._plot_data(self.x_train, self.y_train, 'Train Data')

    def plot_test_data(self):
        self._plot_data(self.x_test, self.y_test, 'Test Data')

    def __repr__(self) -> str:
        return '<BaseTask>'


class SinusoidTask(Task):

    def __init__(self, amplitude: float, phase: float, x, y, test_split: float = 0.2):
        self.amplitude = amplitude
        self.phase = phase
        self.plot_title = f'Sinusoid Task: Amplitude={
            self.amplitude}, Phase={self.phase}'
        super().__init__(x, y, test_split)

    def __repr__(self) -> str:
        return f'<SinusoidTask(amplitude={self.amplitude}, phase={self.phase})>'
