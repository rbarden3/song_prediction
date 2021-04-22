# %%
"""
    usage:
        python print.py path-mpd/
"""
# File Imports
from sorter import alphanum_key
from models import random_forest, knn
from playlists import spotify_conn, get_playlists_from_file, get_track_info, audio_features_df_knn
# from test import audio_features_df_knn

# Package Imports
import sys
import json
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

file_dir = Path(__file__).parent
data_dir = file_dir / 'data'
spotify = spotify_conn(file_dir / 'keys.json')

# %%
# regressor = RandomForestRegressor(warm_start=True)
regressor = RandomForestRegressor()
knn_classifier = KNeighborsClassifier()
for file_path in sorted(data_dir.glob('mpd.slice.*.json'), key=alphanum_key):
    print("fullpath -> ", file_path)
    # %%
    # array_df = get_playlists_from_file(file_path, spotify)

    # # setup_model(array_df)
    # regressor = random_forest(regressor, array_df)
    # print(type(regressor))
    af_df = audio_features_df_knn(file_path, spotify)
    track_pred = knn(af_df)
    # print('predicted track -> ', get_track_info(spotify, track_pred[0]))

    track_name = get_track_info(spotify, track_pred[0])
    print("PREDICTED TRACK NAME -> ", track_name["name"])
    track_artist = get_track_info(spotify, track_pred[0])
    print("PREDICTED ARTIST NAME -> ", track_artist["artists"][0]["name"])
# %%
data_dir = file_dir / 'data'
filenames = os.listdir(data_dir)
