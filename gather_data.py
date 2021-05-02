# %%
"""
    usage:
        python print.py path-mpd/
"""
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
from pathlib import Path
import os
import json
import sys
from get_longest_playlest import get_longest_playlest
from playlists import spotify_conn, get_track_info, audio_features_df_knn
from models import random_forest, knn, sequential_model, build_sequential_model
from sorter import alphanum_key


# File Imports
# from test import audio_features_df_knn

# Package Imports

# with open('all_data.pkl', 'rb') as handle:
#     array_df = pickle.load(handle)

with open('tracks_df.pkl', 'rb') as handle:
    tracks_df = pickle.load(handle)

tracks_df = tracks_df.drop(['artist_name', 'track_name',
                           'type', 'id', 'uri', 'track_href', 'analysis_url'], axis=1)


def get_playlists_from_file(path, conn):
    # Open path to json file, load json data
    data = json.load(open(path))
    dataframe_storage = []
    for ind, playlist in enumerate(data['playlists']):
        # reset track_uri_arr
        tracks = []
        # print("index is ", ind)
        if playlist["tracks"]:
            for track in playlist["tracks"]:
                tracks.append(tracks_df.loc[track["track_uri"]].tolist())
            dataframe_storage.append(pd.DataFrame(tracks))
    return dataframe_storage


file_dir = Path(__file__).parent
data_dir = file_dir / 'data'
spotify = spotify_conn(file_dir / 'keys.json')

#  %%
# regressor = RandomForestRegressor(warm_start=True)
# for file_path in sorted(data_dir.glob('mpd.slice.*.json'), key=alphanum_key):
#     print("fullpath -> ", file_path)
# # %%
#     array_df = get_playlists_from_file(file_path, spotify)

#     # setup_model(array_df)

#     regressor = random_forest(regressor, array_df)

sequential_regressor = build_sequential_model()
regressor = RandomForestRegressor()
knn_classifier = KNeighborsClassifier(n_neighbors=500, p=2, metric='euclidean')
# with open('mpd.slice.0-999.pkl', 'rb') as handle:
#         array_df = pickle.load(handle)
counter = 0
array_df = []
for file_path in sorted(data_dir.glob('mpd.slice.*.json'), key=alphanum_key):
    # if True:
    if counter > 5:
        break

    # file_path = data_dir / 'mpd.slice.0-999.json'
    # print("fullpath -> ", file_path)
    array_df += get_playlists_from_file(file_path, spotify)

    # setup_model(array_df)

    # sequential_regressor = sequential_model(sequential_regressor, array_df)
    # array_df = get_playlists_from_file(file_path, spotify)

    # # setup_model(array_df)
    # regressor = random_forest(regressor, array_df)
    # print(type(regressor))
    # af_df = audio_features_df_knn(file_path, spotify)
    model = knn(knn_classifier, tracks_df)
    # print('predicted track -> ', get_track_info(spotify, track_pred[0]))

    # track_name = get_track_info(spotify, track_pred[0])
    # print("PREDICTED TRACK NAME -> ", track_name["name"])
    # track_artist = get_track_info(spotify, track_pred[0])
    # print("PREDICTED ARTIST NAME -> ", track_artist["artists"][0]["name"])
    counter += 1
# sequential_regressor = sequential_model(sequential_regressor, array_df)
# %%
data_dir = file_dir / 'data'
filenames = os.listdir(data_dir)
