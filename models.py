# File Imports
from data import get_x_y

# Package Imports
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Activation, Lambda
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from playlists import spotify_conn, get_track_info, audio_features_df_knn
from pathlib import Path

# I needed this to allow the use of my GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def random_forest(model, array_of_df):
    dict_x_y = get_x_y(array_of_df)
    X = dict_x_y['x']
    y = dict_x_y['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("y_pred -> ", y_pred)
    mean_error = mean_absolute_error(y_test, y_pred)
    print("\nmean_error -> ", mean_error)
    return model


def build_sequential_model(input_dim=376):
    model = tf.keras.Sequential()
    model.add(LSTM(units=128, input_shape=(
        input_dim, 13), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=13, activation='relu', input_dim=input_dim))
    model.compile(loss="mse", optimizer='adam', metrics=[
                  'accuracy', tf.keras.metrics.MeanSquaredError(), ])
    return model

# def build_sequential_model(input_dim=376):
#     model = tf.keras.Sequential()
#     model.add(LSTM(units=128, input_shape=(
#         input_dim, 13), activation='relu', return_sequences=True))
#     # model.add(Dropout(0.5))
#     model.add(Dropout(0.3))
#     model.add(LSTM(64))
#     model.add(Dense(64))
#     model.add(Dropout(0.3))
#     model.add(Dense(units=13, activation='relu', input_dim=input_dim))
#     model.compile(loss="mse", optimizer='adam', metrics=[
#                   'accuracy', tf.keras.metrics.MeanSquaredError(), ])
#     return model


def sequential_model(model, array_of_df, input_dim=376):
    dict_x_y = get_x_y(array_of_df, average=False, iterate_arrays=False)
    X = dict_x_y['x']
    y = dict_x_y['y']

    # for i, playlist in enumerate(X):
    #     for ind, track in enumerate(playlist):
    #         playlist[ind] = np.asarray(track, dtype=np.float32)
    #     X[i] = np.array(playlist)

    # for ind, song in enumerate(y):
    #     y[ind] = np.asarray(song, dtype=np.float32)

    # X = np.expand_dims(padded, axis=0)
    padded = pad_sequences(X, input_dim)
    X = padded
    # print("Shape X", X.shape)
    y = np.array(y)
    # print("Shape y", y.shape)

    # print("X", X, "\nY:", y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # input_dim = len(X_train[0])

    # model.add(Embedding(376, 256, input_length=input_dim))

    model.fit(X_train, y_train)
    score = model.evaluate(X_test, y_test)
    print(f'Loss: {score[0]} / Accuracy: {score[1]} / MSE: {score[2]}')

    # # Save the model
    # filepath = './saved_model'
    # save_model(model, filepath)
    # # Load the model
    # model = load_model(filepath, compile=True)

    # use_samples = [5, 38, 350]
    # sample_predict = []
    # for sample in use_samples:
    #     sample_predict.append(X_train[sample])

    # sample_predict = np.array(sample_predict)
    # predictions = model.predict(sample_predict)
    # print("Predict:", predictions)

    # y_pred = model.predict(X_test)
    #print("y_pred -> ", y_pred)
    # mean_error = tf.keras.losses.mean_absolute_error(y_test, y_pred)

    #print("\nmean_error -> ", mean_error)
    return model


def get_nth(y_arr):
    if round(math.sqrt(len(y_arr))) % 2 == 0:
        return round(math.sqrt(len(y_arr))) - 1
    else:
        return round(math.sqrt(len(y_arr)))


file_dir = Path(__file__).parent
data_dir = file_dir / 'data'
spotify = spotify_conn(file_dir / 'keys.json')


def knn(model, df):
    df.dropna(inplace=True)
    print(df.columns)
    # split
    X = list(df.T.to_dict('list').values())
    y = list(df.T.to_dict('list').keys())
    # print("y", y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print('Xtrain -> ', X_train)
    # print('X_test -> ', X_test)
    # print('y_train -> ', y_train)
    # print('y_test -> ', y_test)
    # Scaling features all features between -1 and 1
    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)

    # Define and fit model
    nth = get_nth(y_test)
    # classifier = KNeighborsClassifier(n_neighbors=500, p=2, metric='euclidean')
    model.fit(X_train, y_train)

    # Predict
    # y_pred = model.predict(X_test)
    # print("y_pred -> ", y_pred)
    # print("y_pred length", y_pred.shape)

    # track_name = get_track_info(spotify, y_pred[0])
    # print("PREDICTED TRACK NAME -> ", track_name["name"])
    # track_artist = get_track_info(spotify, y_pred[0])
    # print("PREDICTED ARTIST NAME -> ", track_artist["artists"][0]["name"])

    # # Evaluate
    # # cm = confusion_matrix(y_test, y_pred)
    # # print("confusion matrix -> ", cm)
    # print("f1 ->", f1_score(y_test, y_pred, average='micro'))
    # print("accuracy -> ", accuracy_score(y_test, y_pred))
    return model


def xgboost(array_of_df):
    dict_x_y = get_x_y(array_of_df)
    # X = pd.Series(dict_x_y['x'])
    # y = pd.Series(dict_x_y['y'])
    X = pd.DataFrame(dict_x_y['x']).iloc[:, :]
    y = pd.DataFrame(dict_x_y['y']).iloc[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    print("x_train -> ", X_train)
    # regressor = RandomForestRegressor()
    # regressor.fit(X_train, y_train)
    # Train
    # Instantiating a XGBoost classifier object
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    print("x_train", X_train)
    print("y_train", y_train)
    xgb_regressor = xgb.XGBRegressor(
        max_depth=6, learning_rate=0.1, objective='reg:linear', alpha=10, n_estimators=10)
    xgb_regressor.fit(X_train, y_train)

    # Predict
    pred_probs = xgb_regressor.predict_proba(X_test)
    pred_prob_again = xgb_regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_prob_again))

    print("pred_probs ->", pred_probs)

    # Results
    print("y_test -> ", y_test)
    # RMSE
