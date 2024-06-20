import numpy as np
from tasks import SinusoidTask

class TaskGenerator:

    def __init__(self) -> None:
        self.num_tasks = None
        self.num_samples = None

    def generate_tasks(self, num_tasks: int, num_samples: int):
        raise NotImplementedError

class SinusoidTaskGenerator(TaskGenerator):

    def __init__(self,
                 amplitude_range: tuple = (0.1, 5.0),
                 phase_range: tuple = (0, np.pi),
                 x_range: tuple = (-5.0, 5.0)):

        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.x_range = x_range
        super().__init__()

    def generate_tasks(self, num_tasks: int, num_samples: int):
        self.num_tasks = num_tasks
        self.num_samples = num_samples

        tasks = []
        for _ in range(self.num_tasks):
            amplitude = np.random.uniform(*self.amplitude_range)
            phase = np.random.uniform(*self.phase_range)
            x = np.random.uniform(*self.x_range, size=(self.num_samples, 1))
            y = amplitude * np.sin(x - phase)
            task = SinusoidTask(amplitude, phase, x, y)
            tasks.append(task)

        return tasks
