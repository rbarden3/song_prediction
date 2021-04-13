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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Spotify creds
# export SPOTIPY_CLIENT_ID='cf066d284bdd459f9480f4d682555e48'
# export SPOTIPY_CLIENT_SECRET='e7453f0113a042ada35df372168a3481'
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


def get_single_playlist(path):
    # Open path to json file, load json data
    data = json.load(open(path))
    for i in range(len(data["playlists"])):
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


# * Currently working on this.
def setup_model(array_of_df):
    col_labels = get_col_names(array_of_df[0])
    X = pd.DataFrame(array_of_df)
    y = col_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # classifier = RandomForestClassifier(
    #     n_estimators=20, criterion='gini', random_state=1, max_depth=3)
    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # print(y_pred)


if __name__ == "__main__":
    path = sys.argv[1]

    array_df = get_single_playlist(path)
    setup_model(array_df)
