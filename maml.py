from typing import List, Callable

import tensorflow as tf
import numpy as np

from tasks import Task


class MAML:

    def __init__(self,
                 model_generator: Callable,
                 tasks: List[Task],
                 alpha: float = 0.01,
                 beta: float = 0.01,
                 K: int = 5,
                 task_batch_size: int = 10,
                 gradient_steps: int = 10):

        self.model_generator = model_generator
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.task_batch_size = task_batch_size
        self.gradient_steps = gradient_steps
        self.tasks = tasks

        self._train_tasks, self._test_tasks = self._split_tasks()

        self._meta_model = None

    def _split_tasks(self, test_ratio=0.2):

        # Split the tasks into train and test tasks
        num_test_tasks = int(len(self.tasks) * test_ratio)
        test_tasks = self.tasks[:num_test_tasks]
        train_tasks = self.tasks[num_test_tasks:]

        return train_tasks, test_tasks

    def train(self, verbose=True):

        self._meta_model = self.model_generator()

        for step in range(self.gradient_steps):

            # Sample a batch of tasks
            task_batch = np.random.choice(
                self._train_tasks, self.task_batch_size)

            task_losses = []

            for task in task_batch:

                # Clone the model
                base_model = self.model_generator()
                base_model.set_weights(self._meta_model.get_weights())

                # Sample K data points from the task
                x_train_k, y_train_k = task.get_train_data(self.K)

                # Compute the gradient
                with tf.GradientTape() as tape:
                    y_pred = base_model(x_train_k)
                    loss = tf.reduce_mean(tf.square(y_pred - y_train_k))

                gradients = tape.gradient(loss, base_model.trainable_variables)

                # Update the base_model
                base_model.optimizer.apply_gradients(
                    zip(gradients, base_model.trainable_variables))

                # Sample K data points from the task for validation
                x_val_k, y_val_k = task.get_test_data(self.K)

                # Compute the validation loss
                y_pred = base_model(x_val_k)
                task_loss = tf.reduce_mean(tf.square(y_pred - y_val_k))
                task_losses.append(task_loss)

            # Compute the meta gradient using the task losses
            meta_gradients = []
            for meta_weight, task_loss in zip(self._meta_model.get_weights(), task_losses):
                meta_gradients.append(task_loss * meta_weight)

            # Update the meta model
            self._meta_model.optimizer.apply_gradients(
                zip(meta_gradients, self._meta_model.trainable_variables))

            print(f'Step: {step}, Meta Loss: {np.mean(task_losses)}')

        # Evaluate the model on the test tasks
        test_losses = []
        for task in self._test_tasks:
            x_test, y_test = task.get_test_data()
            y_pred = self._meta_model(x_test)
            test_loss = tf.reduce_mean(tf.square(y_pred - y_test))
            test_losses.append(test_loss)

        if verbose:
            print(f'Average Test Loss: {np.mean(test_losses)}')

    def get_model(self):
        return self._meta_model
