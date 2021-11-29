# %%
import pandas as pd
import pickle
# from models import build_sequential_model
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.losses import mean_squared_error
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, Lambda, Flatten
from tensorflow.keras.layers import Embedding, Reshape, GRU, SimpleRNN
from tensorflow.keras.layers import LSTM, Masking, Bidirectional
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import time
from pathlib import Path
from datetime import datetime, date, time
from test_seq_model import rel_error_max
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%%
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

# def rel_error_max(y_true, y_pred):
#     numer = tf.math.abs(tf.math.subtract(y_true, y_pred))
#     denom = tf.math.maximum(tf.math.abs(y_true), tf.math.abs(y_pred))
#     return tf.math.divide(numer, denom)
#%%
def build_sequential_model(input_dim=376):
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    model = tf.keras.Sequential()
    model.add(Masking(0, input_shape=(None, 13)))
    # model.add(Embedding(input_dim=377, output_dim=13))
    model.add(Bidirectional(LSTM(units=52, return_sequences=True)))
    # model.add(Dropout(0.3))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(Bidirectional(GRU(208, return_sequences=True)))
    model.add(Dropout(0.4))

    # model.add(Dropout(0.4, training=True))
    # model.add(Activation('relu'))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    # model.add(Bidirectional(LSTM(units=52, return_sequences=True)))
    # model.add(Dropout(0.3))
    model.add(Bidirectional(SimpleRNN(52)))
    
    model.add(Dropout(0.3))
    # model.add(Dropout(0.3, training=True))
    # model.add(Activation('relu'))

    model.add(Dense(13))
    model.compile(loss='mse', optimizer=opt, metrics=[ tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), rel_error_max])
    return model
#%%
with open('rnn_x_data.pkl', 'rb') as handle:
    x = pickle.load(handle)
with open('rnn_y_data.pkl', 'rb') as handle:
    y = pickle.load(handle)

#%%
x, X_test, y, y_test = train_test_split(x, y, test_size=0.20)
#%%
model_name = 'rnn_history_1000000_D2021-04-28-T20-09'
history_file = model_name+'_samples'+'.pkl'
epoch_file = 'EPOCHSAVE_' + model_name

print("Premodel")
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
model = tf.keras.models.load_model(epoch_file, compile=True)
# model.compile(loss='mse', optimizer=opt, metrics=[ tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), rel_error_max])
# initial_epoch = tf.keras.models.get_init_epoch(epoch_file)
# print()
# print("initial epoch:", initial_epoch)
# print()
# model = tf.keras.saving.saved_model.load(epoch_file)
# model = tf.saved_model.load(epoch_file)
#%%
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5, 
    monitor = 'loss',
    mode='min'
)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=epoch_file,
    save_weights_only=False,
    monitor='val_mean_squared_error',
    mode='min',
    save_best_only=True)

history = model.fit(x = x, y = y, epochs=5,
          validation_data=(X_test, y_test),
          callbacks=[model_checkpoint_callback, early_stopping ])
score = model.evaluate(X_test, y_test)
print(f'Loss: {score[0]} / MSE: {score[1]} / RMSE: {score[2]} / MAE: {score[3]} / rel_err: {score[4]}')
with open(history_file, 'wb') as handle:
    pickle.dump(history.history, handle)
model.save('FINISHEDSAVE_' + model_name)
# %%
