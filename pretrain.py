def pretrain_model(model_generator, tasks):

    # Create a model
    model = model_generator()

    # Pretrain the model on all tasks
    for i, task in enumerate(tasks):
        print("task: ", i, " of ", len(tasks))
        x_train, y_train = task.get_train_data()
        model.fit(x_train, y_train, epochs=1000, verbose=0)
    return model
