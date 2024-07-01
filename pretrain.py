from typing import Callable

def pretrain_model(model_generator: Callable, tasks: list, gradient_steps: int=1000, verbose: int=0):

    # Create a model
    model = model_generator()

    # Pretrain the model on all tasks
    for i, task in enumerate(tasks):

        if verbose:
            print("Task:", i+1, "/", len(tasks))

        x_train, y_train = task.get_train_data()
        model.fit(x_train, y_train, epochs=gradient_steps, verbose=verbose)
        
    return model
