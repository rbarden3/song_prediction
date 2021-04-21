import os
import pandas as pd
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Required for Spotipy
def spotify_conn(keys_path):
    keys = json.load(open(keys_path))
    # Set environment variables
    os.environ['SPOTIPY_CLIENT_ID'] = keys['SPOTIPY_CLIENT_ID']
    os.environ['SPOTIPY_CLIENT_SECRET'] = keys['SPOTIPY_CLIENT_SECRET']
    spotify = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials())
    return spotify

# Given an array of >= 100 tracks, returns the audio features for each track
def get_features(conn, tracks_array):
    return conn.audio_features(tracks_array)

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

def cut_songs_modified(tracks_array):
    req_tracks = []
    while len(tracks_array) > 0 and len(req_tracks) < 100:
        req_tracks.append(tracks_array.pop())
    return (tracks_array, req_tracks)


def get_playlists_from_file(path, conn):
    # Open path to json file, load json data
    data = json.load(open(path))
    dataframe_storage = []
    for ind, playlist in enumerate(data['playlists']):
        # reset track_uri_arr
        track_uri_arr = []
        print("index is ", ind)
        # print(data["playlists"])
        # print("Name -> ", first_playlist["name"])
        # print("Num of Albums -> ", first_playlist["num_albums"])
        # print("Num of Tracks -> ", first_playlist["num_tracks"])
        # print("Tracks -> ")
        for track in playlist["tracks"]:
            track_uri_arr.append(track["track_uri"])

        track_uri_arr = cut_songs(track_uri_arr)
        features_res = get_features(conn, track_uri_arr)
        # time.sleep(1.0)
        new_df = pd.DataFrame(features_res)
        # print(new_df.keys())
        dataframe_storage.append(new_df)
    return dataframe_storage