import matplotlib.pyplot as plt

from maml import MAML
from task_generator import SinusoidTaskGenerator
from models import create_model
from pretrain import pretrain_model

# Generate tasks
task_generator = SinusoidTaskGenerator()
tasks = task_generator.generate_tasks(num_tasks=50, num_samples=100)

# Initialize MAML
maml = MAML(model_generator=create_model, tasks=tasks)

# Train MAML
maml.train(verbose=True)
meta_model = maml.get_model()

# Generate a task for few-shot learning
few_shot_task = task_generator.generate_tasks(num_tasks=1, num_samples=10000)[0]

# Perform few-shot learning
x_train, y_train = few_shot_task.get_train_data(K=5)

# Clone the model
few_shot_model = create_model()
few_shot_model.set_weights(meta_model.get_weights())

# Train the few-shot meta-learned model
few_shot_model.fit(x_train, y_train, epochs=10)

# Evaluate the few-shot meta-learned model
x_test, y_test = few_shot_task.get_test_data()
loss = few_shot_model.evaluate(x_test, y_test, verbose=0)
print(f'Few-shot loss: {loss}')

# Create a model for comparison
pretrained_model = pretrain_model(create_model, tasks)

# Train the pretrained model on the few-shot task
pretrained_model.fit(x_train, y_train, epochs=1)

# Evaluate the pretrained model
loss = pretrained_model.evaluate(x_test, y_test, verbose=0)
print(f'Pretrained model loss: {loss}')


# Plot the predictions for the few-shot task test data
y_pred_few_shot = few_shot_model.predict(x_test)
y_pred_pretrained = pretrained_model.predict(x_test)

plt.scatter(x_test, y_test, label='Ground Truth')
plt.scatter(x_test, y_pred_few_shot, label='MAML')
plt.scatter(x_test, y_pred_pretrained, label='Pretrained')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
