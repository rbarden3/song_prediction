# %%
"""
    usage:
        python print.py path-mpd/
"""
import sys
import json
import time
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

file_dir = Path(__file__).parent
keys = json.load(open(file_dir / 'keys.json'))
# Set environment variables
os.environ['SPOTIPY_CLIENT_ID'] = keys['SPOTIPY_CLIENT_ID']
os.environ['SPOTIPY_CLIENT_SECRET'] = keys['SPOTIPY_CLIENT_SECRET']

# Required for Spotipy
spotify = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials())

track_uri_arr = []
dataframe_storage = []

# converts to a df given a list


def to_dataframe(array_of_features):
    return pd.DataFrame(data=array_of_features)

# Given an array of >= 100 tracks, returns the audio features for each track


def get_features(tracks_array):
    return spotify.audio_features(tracks_array)

# Get 0 index playlist
# Like process_playlists, but just processes the first [0] playlist


def cut_songs(tracks_array):
    # the tracks_array inputted should be > 100 songs. this will just trim
    # off songs until the array = 100 songs
    if len(tracks_array) > 100:
        items_to_delete = len(tracks_array) - 100
        del tracks_array[len(tracks_array) - items_to_delete:]
        return tracks_array
    else:
        return tracks_array


def get_col_names(dataframe_df):
    df_list = []
    for col in dataframe_df.columns:
        df_list.append(col)
    return df_list


def get_playlists_from_file(path):
    # Open path to json file, load json data
    data = json.load(open(path))
    # for i in range(len(data["playlists"])):
    for i in range(len(data['playlists'])):
        # reset track_uri_arr
        track_uri_arr = []
        ith_playlist = data['playlists'][i]
        print("index is ", i)
        # print(data["playlists"])
        # print("Name -> ", first_playlist["name"])
        # print("Num of Albums -> ", first_playlist["num_albums"])
        # print("Num of Tracks -> ", first_playlist["num_tracks"])
        # print("Tracks -> ")
        for track in ith_playlist["tracks"]:
            track_uri_arr.append(track["track_uri"])

        track_uri_arr = cut_songs(track_uri_arr)
        features_res = get_features(track_uri_arr)
        # time.sleep(1.0)
        new_df = to_dataframe(features_res)
        # print(new_df.keys())
        dataframe_storage.append(new_df)
    return dataframe_storage
# %%
# * Currently working on this.


def setup_rf_model(model, array_of_df):
    dict_x_y = get_x_y(array_of_df)
    X = dict_x_y['x']
    y = dict_x_y['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("y_pred -> ", y_pred)
    mean_error = metrics.mean_absolute_error(y_test, y_pred)
    print("\nmean_error -> ", mean_error)
    return model
# %%
# * Currently working on this.


def setup_model(array_of_df):
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
    pred_prob_again = xgb_regressor.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_prob_again))

    print("pred_probs ->", pred_probs)

    # Results
    print("y_test -> ", y_test)
    # RMSE
# %%


def split_x_y(in_df):
    # Col names
    # ['acousticness', 'analysis_url', 'danceability', 'duration_ms', 'energy', 'id', 'instrumentalness',
    # 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'track_href', 'type', 'uri', 'valence']
    in_df = in_df.drop(['analysis_url', 'track_href',
                       'uri', 'id', 'type'], axis='columns')
    in_df, last_row = in_df.drop(in_df.tail(1).index), in_df.tail(1)
    x = in_df.mean().tolist()
    y = last_row.values.tolist()[0]

    return {'x': x, 'y': y}


def get_x_y(in_df_arr):
    out = {'x': [], 'y': []}
    # print(in_df_arr)
    for _, val in enumerate(in_df_arr):
        #print("val -> ", val)
        split_data = split_x_y(val)
        out['x'].append(split_data['x'])
        out['y'].append(split_data['y'])
    return out


def split_df_array(arr_df):
    features_arr = []
    for i in range(len(arr_df)):
        features_arr.append(split_x_y(arr_df[i]))

    return features_arr


# %%
# try:
#     path = sys.argv[1]
# except IndexError:
# from_ = 883000
# upto = 883999
for i in range(100):
   # build_string = "mpd.slice." + str(from_) + "-" + str(upto) + ".json"
    #path = file_dir / 'data' / build_string
    #print("path -> ", path)
    pth_data = file_dir / 'data'
    filenames = os.listdir(pth_data)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((str(pth_data), filename))
            print("fullpath -> ", fullpath)
            # %%
            array_df = get_playlists_from_file(fullpath)
            # %%
            # setup_model(array_df)

            # %%
            regressor = RandomForestRegressor()
            #regressor = setup_rf_model(regressor, array_df)
            regressor = setup_rf_model(regressor, array_df)

            # from_ += 1000
            # upto += 1000
            # if from_ > 999000:
            #     break
            # %%
