import json

import matplotlib.pyplot as plt

from maml import MAML
from task_generator import SinusoidTaskGenerator
from models import create_model
from pretrain import pretrain_model
from utils import Range

with open('config.json', encoding="utf-8") as f:
    config = json.load(f)

# Generate tasks
task_generator = SinusoidTaskGenerator(
    amplitude_range=Range(**config['sinusoid_config']['amplitude_range']),
    phase_range=Range(**config['sinusoid_config']['phase_range']),
    x_range=Range(**config['sinusoid_config']['x_range']),
    test_split=config['task_config']['test_split']
)
tasks = task_generator.generate_tasks(
    num_tasks=config['task_config']['train_tasks'],
    num_samples=config['task_config']['train_tasks_samples']
)
# Initialize MAML
maml = MAML(
    model_generator=create_model,
    tasks=tasks,
    alpha=config['maml_config']['alpha'],
    beta=config['maml_config']['beta'],
    K=config['maml_config']['K'],
    task_batch_size=config['maml_config']['task_batch_size'],
    gradient_steps=config['maml_config']['gradient_steps'],
    test_ratio=config['task_config']['test_split']
)

# Train MAML
maml.train(verbose=True)
meta_model = maml.get_model()

# Generate a task for few-shot learning
few_shot_task = task_generator.generate_tasks(
    num_tasks=1, num_samples=config['task_config']['few_shot_task_samples'])[0]

# Perform few-shot learning
x_train, y_train = few_shot_task.get_train_data(K=config['maml_config']['K'])

# Train the few-shot meta-learned model
meta_model.fit(x_train, y_train, epochs=10)

# Evaluate the few-shot meta-learned model
x_test, y_test = few_shot_task.get_test_data()
loss = meta_model.evaluate(x_test, y_test, verbose=0)
print(f'Meta Model loss: {loss}')

# Create a model for comparison
pretrained_model = pretrain_model(create_model, tasks)

# Train the pretrained model on the few-shot task
pretrained_model.fit(x_train, y_train, epochs=1)

# Evaluate the pretrained model
loss = pretrained_model.evaluate(x_test, y_test, verbose=0)
print(f'Pretrained Model loss: {loss}')


# Plot the predictions for the few-shot task test data
y_pred_few_shot = meta_model.predict(x_test)
y_pred_pretrained = pretrained_model.predict(x_test)

plt.scatter(x_test, y_test, label='Ground Truth')
plt.scatter(x_test, y_pred_few_shot, label='MAML')
plt.scatter(x_test, y_pred_pretrained, label='Pretrained')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
