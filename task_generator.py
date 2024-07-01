from abc import ABC, abstractmethod

import numpy as np

from tasks import SinusoidTask
from utils import Range


class TaskGenerator(ABC):

    def __init__(self) -> None:
        self.num_tasks = None
        self.num_samples = None

    @abstractmethod
    def generate_tasks(self, num_tasks: int, num_samples: int):
        raise NotImplementedError


class SinusoidTaskGenerator(TaskGenerator):

    def __init__(self,
                 amplitude_range: Range = Range(0.1, 5.0),
                 phase_range: Range = Range(0, np.pi),
                 x_range: Range = Range(-5.0, 5.0),
                 test_split: float = 0.2):

        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.x_range = x_range
        self.test_split = test_split
        super().__init__()

    def generate_tasks(self, num_tasks: int, num_samples: int):

        self.num_tasks = num_tasks
        self.num_samples = num_samples

        tasks = []
        for _ in range(self.num_tasks):
            # Use the min and max attributes of the Range instances
            amplitude = np.random.uniform(
                self.amplitude_range.min, self.amplitude_range.max)
            phase = np.random.uniform(
                self.phase_range.min, self.phase_range.max)

            x = np.random.uniform(
                self.x_range.min, self.x_range.max, size=(self.num_samples, 1))
            y = amplitude * np.sin(x - phase)

            task = SinusoidTask(x=x, y=y, amplitude=amplitude,
                                phase=phase, test_split=self.test_split)
            tasks.append(task)

        return tasks
