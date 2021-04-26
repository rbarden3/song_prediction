# %%
import pandas as pd
import pickle
# from models import build_sequential_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, Lambda, Flatten, Reshape
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
#%%
# max_features = 20000
# batch_size = 32
# BUFFER_SIZE=1000

# (x_train, y_train), (x_test, y_test)=tf.keras.datasets.imdb.load_data(
#     path="imdb.npz",
#     num_words=max_features,
#     skip_top=0,
#     maxlen=None,
#     seed=113,
#     start_char=1,
#     oov_char=2,
#     index_from=3)
#%%
def build_sequential_model():
    model = tf.keras.Sequential()
    model.add(Input(shape=(None, 13)))
    # model.add(Reshape((42, 13))) 
    model.add(Embedding(376, 13)) 
    model.add(LSTM(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=13, activation='relu'))
    model.compile(loss="mse", optimizer='adam', metrics=['accuracy', tf.keras.metrics.MeanSquaredError(), ])
    return model
    

#%%
# with open('all_data.pkl', 'rb') as handle:
#     array_df = pickle.load(handle)

print("\nData Loaded...\n")
#%%
model = build_sequential_model()
print("\nModel Built...\n")
#%%
# sequential_regressor = sequential_model(sequential_regressor, array_df)
input_dim = 376
x = []
y = []
test = True
if test:
    with open('mpd.slice.0-999.pkl', 'rb') as handle:
        array_df = pickle.load(handle)
        # print(array_df[0].columns)
    for df in array_df:
        df = df.drop(['type', 'id', 'uri', 'track_href', 'analysis_url'], axis = 1)
        # y.append(tf.constant(df.tail(1).values.tolist(), shape=(1, 13), dtype=np.float32))
        x_list = df.drop(df.tail(1).index).values.tolist()
        y_list = df.tail(1).values.tolist()[0]
        # x.append(tf.constant(x_list, shape=(len(x_list), 13), dtype=np.float32))
        # x_list  = pad_sequences(x_list)
        # x.append(np.asarray(x_list, dtype=np.float32))
        # y.append(np.asarray(df.tail(1).values.tolist()[0], dtype=np.float32)).
        # np_x = np.asarray(x_list)
        # np_y = np.asarray(y_list)
        np_x = np.asarray(x_list)
        np_y = np.asarray(y_list)
        # np_x = np_x.reshape((1, np_x.shape[0], 13 ))
        x.append(np_x)
        y.append(np_y)
        # x.append(x_list)
        # y.append(y_list)

# dataset = tf.data.Dataset.from_tensor_slices((x, y))
# del array_df
# x = np.asarray(x).reshape(-1,1)

print("\nData Formatted...\n")
# padded = pad_sequences(x, input_dim)
# X=tf.ragged.constant(x)
# X = padded
# y = y
# X = x
# y = np.asarray(y)
# X = np.asarray(x)
# y = tf.constant(y)
# X = tf.ragged.constant(x)

#%%
# print("X", X, "\nY:", y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
#%%
# y_train=np.asarray(y_train)
# y_test=np.asarray(y_test)
# X_train=np.asarray(X_train)
# X_test=np.asarray(X_test)
# y_train=tf.constant(y_train)
# y_test=tf.constant(y_test)

for ind, playlist in enumerate(X_train):
    # X_train[ind]=tf.constant(np.asarray(playlist))
    X_train[ind]=tf.constant(playlist)

    # shape = list(X_train[ind].shape)
    # padding = [[0 for _ in range(shape[0])],[0 for _ in range(shape[1])]]
    # X_train[ind]=tf.space_to_batch(X_train[ind], block_shape=shape, paddings=padding)
    # X_train[ind]=np.asarray(playlist)
for ind, playlist in enumerate(X_test):
    # X_test[ind]=tf.constant(np.asarray(playlist))
    X_test[ind]=tf.constant(playlist)
    # shape = list(X_test[ind].shape)
    # padding = [[0 for _ in range(shape[0])],[0 for _ in range(shape[1])]]
    # X_test[ind]=tf.space_to_batch(X_test[ind], block_shape=shape, paddings=padding)
    # X_test[ind]=np.asarray(playlist)
#%%
# x_train=tf.ragged.constant(X_train)
# x_test=tf.ragged.constant(X_test)

# x_train=tf.ragged.stack(X_train, name='x_train')
# x_test=tf.ragged.stack(X_test, name='x_test')
# x_train=np.asarray(X_train,dtype=object)
# x_test=np.asarray(X_test,dtype=object)

# x_train=tf.ragged.stack(X_train, name='x_train').to_tensor()
# x_test=tf.ragged.stack(X_test, name='x_test').to_tensor()

# x_train=tf.reshape(x_train, (-1,1))
# x_test=tf.reshape(x_test, (-1,1))

# input_dim = len(X_train[0])
#%%
# model.add(Embedding(376, 256, input_length=input_dim))
# train_data = [(X_train[i], y_train[i]) for i in range(0, len(X_train))]
#%%
# for train in train_data:
print("\nTraining Model...\n")
model.fit(x = np.asarray(X_train), y = y_train)
print("\nModel Trained...\n")
score = model.evaluate(X_test, y_test)
print(f'Loss: {score[0]} / Accuracy: {score[1]} / MSE: {score[2]}')

# %%
