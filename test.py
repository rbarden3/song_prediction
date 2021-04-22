# %%
# from playlists import get_features, cut_songs_modified, spotify_conn
# from pathlib import Path
# import pandas as pd
# import json
# %%


# def cut_songs_dict(tracks: dict):
#     request_tracks = dict()
#     while len(tracks) > 0 and len(request_tracks) < 100:
#         uri, features = tracks.popitem()
#         request_tracks[uri] = features
#     return (tracks, request_tracks)


# # %%
# file_dir = Path(__file__).parent
# data_dir = file_dir / 'data'
# conn = spotify_conn(file_dir / 'keys.json')
# path = data_dir / 'mpd.slice.8000-8999.json'
# # %%


# def audio_features_df_knn(path, conn):
#     data = json.load(open(path))
#     field_names = ["artist_name", "track_name"]
#     file_tracks = dict()
#     track_uri_dict = dict()
#     for ind, playlist in enumerate(data['playlists']):
#         for track in playlist["tracks"]:
#             track_uri_dict[track["track_uri"]] = [
#                 track["artist_name"], track["track_name"]]
#     counter = 0
#     while(len(track_uri_dict) > 0):
#         counter += 1
#         track_uri_dict, request_tracks = cut_songs_dict(track_uri_dict)
#         track_uris = request_tracks.keys()
#         features_res = get_features(conn, track_uris)
#         for ind, val in enumerate(features_res):
#             request_tracks[val['uri']] = request_tracks[val['uri']
#                                                         ] + list(val.values())
#         file_tracks.update(request_tracks)
#     field_names += val.keys()

#     df = pd.DataFrame.from_dict(
#         data=file_tracks, orient='index', columns=field_names)
#     df = df.drop(['analysis_url', 'track_href',
#                   'uri', 'id', 'type'], axis='columns')
#     print(df)
#     return df
from playlists import get_features, get_all_tracks, spotify_conn
from pathlib import Path
import pandas as pd
import pickle
import json
from tqdm import tqdm
import spotipy
import time
import requests
from sorter import alphanum_key

# %%


def cut_songs_dict(tracks: dict):
    request_tracks = dict()
    while len(tracks) > 0 and len(request_tracks) < 100:
        uri, features = tracks.popitem()
        request_tracks[uri] = features
    return (tracks, request_tracks)


# %%
file_dir = Path(__file__).parent
data_dir = file_dir / 'data'
pkl_path = file_dir / 'tracks_dict_noFeatures.pkl'
conn = spotify_conn(file_dir / 'keys.json')\

# path = data_dir / 'mpd.slice.0-999.json'
# path = data_dir / 'mpd.slice.8000-8999.json'
# %%
all_tracks_noFeatures = dict()
all_tracks = dict()
time_start = time.time()
request_times = []

try:
    with open(pkl_path, 'rb') as handle:
        all_tracks_noFeatures = pickle.load(handle)
except:
    all_tracks_noFeatures = get_all_tracks(data_dir)
    with open(pkl_path, 'wb') as handle:
        pickle.dump(all_tracks, handle)
base_length = len(all_tracks_noFeatures)
percent_complete = int(base_length - len(all_tracks_noFeatures) / base_length)
# %%
counter = 0
# print("Loops Completed: " , end ="" )
pbar = tqdm(total=int(base_length/100)+1)
while(len(all_tracks_noFeatures) > 0):
    # print(counter , end ="," )
    # print(round((base_length - len(all_tracks_noFeatures)) / base_length,2) , end ="," )

    # current_percent= round((base_length - len(all_tracks_noFeatures)) / base_length,2)
    # if(percent_complete != current_percent):
    #     percent_complete = current_percent
    #     print(str(percent_complete) + "% Complete")
    request_start = time.time()
    all_tracks_noFeatures, request_tracks = cut_songs_dict(
        all_tracks_noFeatures)
    track_uris = request_tracks.keys()
    count = 0
    while count < 10:
        try:
            features_res = get_features(conn, track_uris)
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
        except  spotipy.SpotifyException as e:
            # print(e)
            count +=1
            if count < 20:
                time.sleep(15)
                pass
            else:
                raise(e)
    for ind, val in enumerate(features_res):
        try:
            request_tracks[val['uri']
                           ] = request_tracks[val['uri']] + list(val.values())
        except TypeError as e:
            # print("Error Assigning URI, Song Removed from Sporify:", e)
            pass
    all_tracks.update(request_tracks)
    request_end = time.time()
    request_times.append(request_end - request_start)
    # print("Request completed in: " + str(request_end - request_start))
    counter += 1
    pbar.update(1)
pbar.close()
field_names = ["artist_name", "track_name"] + list(val.keys())
df = pd.DataFrame.from_dict(
    data=all_tracks, orient='index', columns=field_names)
# print("results compiled in: " + str(time.time()-time_start))
# print("Average Time: " + str(sum(request_times)/len(request_times)))
df.to_pickle('tracks_df.pkl')
#%%
# df2 = pd.read_pickle('tracks_df.pkl')
