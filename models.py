from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    model = Sequential()
    model.add(Dense(40, input_shape=(1,), activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model