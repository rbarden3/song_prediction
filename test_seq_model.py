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
from tensorflow.keras.layers import LSTM, Masking
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

# def build_sequential_model(input_dim=376):
#     model = tf.keras.Sequential()
#     # model.add(Embedding(input_dim=377, output_dim=13))
#     model.add(LSTM(units=512, input_shape=(None, 13), activation='tanh', return_sequences=True))
#     model.add(Dropout(0.3))

#     # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
#     model.add(GRU(256, return_sequences=True))
#     model.add(Dropout(0.5))
#     model.add(Activation('relu'))

#     # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
#     model.add(SimpleRNN(128))
#     model.add(Dropout(0.3))
#     model.add(Activation('relu'))

#     model.add(Dense(13))
#     model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=[
#                   'accuracy', tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), ])
#     return model

def build_sequential_model(input_dim=376):
    model = tf.keras.Sequential()
    model.add(Masking(0, input_shape=(None, 13)))
    # model.add(Embedding(input_dim=377, output_dim=13))
    model.add(LSTM(units=52, activation='tanh', return_sequences=True))
    model.add(Dropout(0.3))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(GRU(208, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(SimpleRNN(52))
    model.add(Dropout(0.3))
    model.add(Activation('relu'))

    model.add(Dense(13))
    model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=[
                  'accuracy', tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), ])
    return model


#%%
model = build_sequential_model()
#%%

with open('rnn_x_dataTEST.pkl', 'rb') as handle:
    x = pickle.load(handle)
with open('rnn_y_dataTEST.pkl', 'rb') as handle:
    y = pickle.load(handle)
# with open('rnn_x_data.pkl', 'rb') as handle:
#     x = pickle.load(handle)
# with open('rnn_y_data.pkl', 'rb') as handle:
#     y = pickle.load(handle)

#%%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
#%%
model.fit(x = X_train, y = y_train, epochs=5)
score = model.evaluate(X_test, y_test)
print(f'Loss: {score[0]} / Accuracy: {score[1]} / MSE: {score[2]} / MAE: {score[3]}')
# img_file = './model_arch.png'
# tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)
# tf.keras.utils.plot_model(model, show_shapes=True)
# %%
