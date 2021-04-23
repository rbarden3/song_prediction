import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

num_epochs = 5

def deep_model():
    model = Sequential()
    model.add(
        Dense(units=13, activation='relu', input_dim=13))
    model.add(
        Dense(units=6, activation='relu'))
    model.add(
        Dense(units=3, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=num_epochs, batch_size=16, verbose=0)

    preds = model.predict(X_test)
    error = preds - y_test
    error = np.absolute(error)
    best_mean_error = np.average(error)

    return (best_mean_error, model)