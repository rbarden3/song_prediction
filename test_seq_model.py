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

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# def build_sequential_model():
#     model = tf.keras.Sequential()
#     model.add(Input(shape=(None, 13)))
#     model.add(Reshape((42, 13))) 
#     # model.add(Embedding(800, 13)) 
#     model.add(LSTM(units=128, batch_input_shape=(1, None, 13), activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(units=13, activation='relu'))
#     model.compile(loss="mse", optimizer='adam', metrics=['accuracy', tf.keras.metrics.MeanSquaredError(), ])
#     return model

# def build_sequential_model(input_dim=376):
#     model = tf.keras.Sequential()
#     # model.add(LSTM(units=128, input_shape=(None, 13), activation='relu'))
#     model.add(LSTM(units=128, input_shape=(None, 13), activation='relu'))

#     model.add(Dropout(0.5))
#     model.add(Dense(units=13, activation='relu'))
#     model.compile(loss="mse", optimizer='adam', metrics=[
#                   'accuracy', tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), ])
#     return model

# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(mean_squared_error(y_true, y_pred))

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
# print("\nModel Built...\n")
#%%
# sequential_regressor = sequential_model(sequential_regressor, array_df)

# with open('all_data.pkl', 'rb') as handle:
#     array_df = pickle.load(handle)

# print("\nData Loaded...\n")
# #%%
# input_dim = 376
x = []
y = []
# test = False
# if test:
#     with open('mpd.slice.0-999.pkl', 'rb') as handle:
#         array_df = pickle.load(handle)
#         # print(array_df[0].columns)
#     for df in array_df:
#         df = df.drop(['type', 'id', 'uri', 'track_href', 'analysis_url'], axis = 1)
#         # y.append(tf.constant(df.tail(1).values.tolist(), shape=(1, 13), dtype=np.float32))
#         y_list = df.tail(1).values.tolist()[0]
#         x_list = df.drop(df.tail(1).index).values.tolist()
#         x.append(x_list)
#         y.append(y_list)
# else:
#     with open('all_data.pkl', 'rb') as handle:
#         array_df = pickle.load(handle)
#     # for df in array_df:
#     #     x.append(df.drop(df.tail(1).index).values.tolist())
#     #     y.append(df.tail(1).values.tolist()[0])

#     # playlists = len(array_df)
#     playlists = 50000
#     for _ in tqdm(range(playlists)):
#         df = array_df.pop()
#         df.dropna(inplace=True)
#         if df.shape[0] > 0:
#             y_list = df.tail(1).values.tolist()[0]
#             x_list = df.drop(df.tail(1).index).values.tolist()
#             y.append(y_list)
#             x.append(x_list)

#%%
# print("\nData Formatted...\n")
# df.tail(1).values.tolist()[0]
# padded = pad_sequences(x)
# X = padded
#%%
# from write_train_test_data import get_model_xy
# x, y = get_model_xy(playlist_count=50000)  
# x = pad_sequences(x)
# y = np.asarray(y)

# with open('rnn_x_dataTEST.pkl', 'wb') as handle:
#     pickle.dump(x, handle)
# with open('rnn_y_dataTEST.pkl', 'wb') as handle:
#     pickle.dump(y, handle)


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
# print("\nTraining Model...\n")
model.fit(x = X_train, y = y_train, epochs=5)
# print("\nModel Trained...\n")
score = model.evaluate(X_test, y_test)
print(f'Loss: {score[0]} / Accuracy: {score[1]} / MSE: {score[2]} / MAE: {score[3]}')
# img_file = './model_arch.png'
# tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)
# tf.keras.utils.plot_model(model, show_shapes=True)
# %%
