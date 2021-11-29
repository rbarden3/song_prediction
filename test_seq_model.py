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

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#%%
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

def rel_error_max(y_true, y_pred):
    numer = tf.math.abs(tf.math.subtract(y_true, y_pred))
    denom = tf.math.maximum(tf.math.abs(y_true), tf.math.abs(y_pred))
    return tf.math.divide(numer, denom)

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
    model.compile(loss='mse', optimizer=opt, metrics=[ tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    return model
#%%
# def build_sequential_model(input_dim=376):
#     opt = tf.keras.optimizers.Adam(learning_rate=0.1)
#     model = tf.keras.Sequential()
#     model.add(Masking(0, input_shape=(None, 13)))
#     # model.add(Embedding(input_dim=377, output_dim=13))
#     model.add(LSTM(units=13, activation='tanh', return_sequences=True))
#     model.add(Dropout(0.3))

#     # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
#     model.add(GRU(52, return_sequences=True))
#     model.add(Dropout(0.5))
#     model.add(Activation('relu'))

#     # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
#     model.add(SimpleRNN(13))
#     model.add(Dropout(0.3))
#     model.add(Activation('relu'))

#     model.add(Dense(13))
#     model.compile(loss=root_mean_squared_error, optimizer=opt, metrics=[ tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), ])
#     return model

#%%
# model = build_sequential_model()
#%%

with open('rnn_x_data.pkl', 'rb') as handle:
    x = pickle.load(handle)
with open('rnn_y_data.pkl', 'rb') as handle:
    y = pickle.load(handle)
# x = x[:10000]
# y = y[:10000]
# with open('rnn_x_data.pkl', 'rb') as handle:
#     x = pickle.load(handle)
# with open('rnn_y_data.pkl', 'rb') as handle:
#     y = pickle.load(handle)


# %%
def compare_feats(in_model, in_x, in_y):
    cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    preds = model.predict(in_x)
    errors = []
    for ind, song in enumerate(in_y):
        vals = {'y':song, 'y_pred':preds[ind],'MSE':[], 'percent_errors':[], 'percent_errors_max':[], 'rel_errors':[], 'rel_error_max':[]}
        for i, feat in enumerate(song):
            pred_feat = preds[ind][i]
            vals['MSE'].append((feat-pred_feat)**2) 
            vals['percent_errors'].append(abs(pred_feat-feat)/abs(feat)) 
            vals['percent_errors_max'].append(abs(pred_feat-feat)/max(0.000001,abs(feat)))
            vals['rel_errors'].append(abs(feat-pred_feat)/((abs(feat)+abs(pred_feat))/2))
            vals['rel_error_max'].append(abs(feat-pred_feat)/(max(abs(feat),abs(pred_feat))))
            # vals['rel_error_max'].append(abs(feat-pred_feat)/(max(abs(feat),abs(pred_feat))))
            

        errors.append(pd.DataFrame(list(vals.values()), list(vals.keys()), cols))
    return errors

#%%
# with open('rnn_history_20000_samples.pkl', 'rb') as handle:
#     history_dict = pickle.load(handle)
# %%
#%%
x, X_test, y, y_test = train_test_split(x, y, test_size=0.20)
#%%
model_name = 'rnn_history_'+ str(len(x))+'_D'+datetime.now().strftime('%Y-%m-%d-T%H-%M')
history_file = model_name+'_samples_'+'.pkl'

model = build_sequential_model()
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5, 
    mode='min'
)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='EPOCHSAVE_' + model_name,
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
model.save(model_name)
# img_file = './model_arch.png'
# tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)
# tf.keras.utils.plot_model(model, show_shapes=True)
# %%
import plotly.graph_objects as go
history_dict = history.history
fig = go.Figure()
for key in history_dict.keys():
    fig.add_trace(go.Scatter(
        x=list(range(len(history_dict[key]))),
        y=history_dict[key],
        showlegend=True,
        name=key,
    ))
fig.update_traces(mode='lines')
fig.show()