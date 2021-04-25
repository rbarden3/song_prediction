# %%
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


with open('tracks_df.pkl', 'rb') as handle:
    tracks_df = pickle.load(handle)

tracks_df = tracks_df.drop(['artist_name', 'track_name',
                           'type', 'id', 'uri', 'track_href', 'analysis_url'], axis=1)

file_dir = Path(__file__).parent
data_dir = file_dir / 'data'
spotify = spotify_conn(file_dir / 'keys.json')


knn_classifier = KNeighborsClassifier()
# with open('mpd.slice.0-999.pkl', 'rb') as handle:
#         array_df = pickle.load(handle)
counter = 0
array_df = []

track_pred = knn(tracks_df)
print('predicted track -> ', get_track_info(spotify, track_pred[0]))

track_name = get_track_info(spotify, track_pred[0])
print("PREDICTED TRACK NAME -> ", track_name["name"])
track_artist = get_track_info(spotify, track_pred[0])
print("PREDICTED ARTIST NAME -> ", track_artist["artists"][0]["name"])
