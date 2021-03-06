# %%
import os
import pandas as pd
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
from pathlib import Path
from sorter import alphanum_key
import requests

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
    count = 0
    while count < 10:
        try:
            features_res = conn.audio_features(tracks_array)
            count = 11
        except requests.exceptions.ReadTimeout as e:
            # print(e)
            count += 1
            if count < 10:
                # print("Get Failed w/ ReadTimeout error: ", e)
                # print("Trying Again: Attempt", count+1, "out of 10")
                pass
            else:
                # print("Get Failed after 10 attempts")
                raise(e)
        except spotipy.SpotifyException as e:
            # print(e)
            count += 1
            if count < 20:
                time.sleep(15)
                pass
            else:
                raise(e)
    return features_res


def get_track_info(conn, track_id):
    return conn.track(track_id)
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

# we can probably delete this, it's not used


def cut_songs_modified(tracks_array):
    if len(tracks_array) > 100:
        items = 100
    else:
        items = len(tracks_array)
    req_tracks = tracks_array[:items]
    del tracks_array[:items]
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
        features_res = []
        for track in playlist["tracks"]:
            track_uri_arr.append(track["track_uri"])
        while len(track_uri_arr) > 0:
            track_uri_arr, req_tracks = cut_songs_modified(track_uri_arr)
            features_res += get_features(conn, req_tracks)
        dataframe_storage.append(pd.DataFrame(features_res))
    return dataframe_storage


def get_playlists_from_file_NoFeats(path):
    # Open path to json file, load json data
    data = json.load(open(path))
    all_playlists = []
    for playlist in data['playlists']:
        all_playlists.append(playlist["tracks"])
    return all_playlists


def cut_songs_dict(tracks: dict):
    request_tracks = dict()
    while len(tracks) > 0 and len(request_tracks) < 100:
        uri, features = tracks.popitem()
        request_tracks[uri] = features
    return (tracks, request_tracks)


def audio_features_df_knn(path, conn):
    data = json.load(open(path))
    field_names = ["artist_name", "track_name"]
    file_tracks = dict()
    track_uri_dict = dict()
    for ind, playlist in enumerate(data['playlists']):
        for track in playlist["tracks"]:
            track_uri_dict[track["track_uri"]] = [
                track["artist_name"], track["track_name"]]
    counter = 0
    while(len(track_uri_dict) > 0):
        counter += 1
        track_uri_dict, request_tracks = cut_songs_dict(track_uri_dict)
        track_uris = request_tracks.keys()
        features_res = get_features(conn, track_uris)
        for ind, val in enumerate(features_res):
            request_tracks[val['uri']] = request_tracks[val['uri']
                                                        ] + list(val.values())
        file_tracks.update(request_tracks)
    field_names += val.keys()

    df = pd.DataFrame.from_dict(
        data=file_tracks, orient='index', columns=field_names)
    try:
        df = df.drop(['analysis_url', 'track_href',
                      'uri', 'id', 'type'], axis='columns')
    except KeyError:
        pass
    print(df)
    return df


def get_all_tracks(data_dir):
    all_tracks = dict()
    time_start = time.time()
    file_times = dict()
    # file_path = data_dir / 'mpd.slice.21000-21999.json'
    for file_path in sorted(data_dir.glob('mpd.slice.*.json'), key=alphanum_key):
        # if True:
        file_start = time.time()
        print("file path -> ", file_path)
        data = json.load(open(file_path))
        files_tracks = dict()
        for ind, playlist in enumerate(data['playlists']):
            for track in playlist["tracks"]:
                files_tracks[track["track_uri"]] = [
                    track["artist_name"], track["track_name"]]
        all_tracks.update(files_tracks)

        file_end = time.time()
        file_times[file_path] = file_end - file_start

        print("file completed in: " + str(file_times[file_path]))
    print("results compiled in: " + str(time.time()-time_start))
    print("Average Time: " + str(sum(file_times.values())/len(file_times)))
    return all_tracks
