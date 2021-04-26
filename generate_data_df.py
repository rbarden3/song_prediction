#%%
from data import get_x_y
from pathlib import Path
from playlists import spotify_conn #, get_playlists_from_file
import numpy as np
import json
from sorter import alphanum_key
# import pickle
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm
# from data import get_features
from playlists import cut_songs_modified
#%%
file_dir = Path(__file__).parent
data_dir = file_dir / 'data'
spotify = spotify_conn(file_dir / 'keys.json')
#%%
with open('tracks_df.pkl', 'rb') as handle:
    tracks_df = pickle.load(handle)

tracks_df = tracks_df.drop(['artist_name', 'track_name', 'type', 'id', 'uri', 'track_href', 'analysis_url'], axis = 1)
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

#%%
array_df = []
for file_path in tqdm(sorted(data_dir.glob('mpd.slice.*.json'), key=alphanum_key)):
    # print("File:", file_path)
    file_dfs = get_playlists_from_file(file_path, spotify)
    array_df += file_dfs

with open('all_data_noNA.pkl', 'wb') as handle:
    pickle.dump(array_df, handle)
with open('all_data_noNA.pkl', 'rb') as handle:
    read_array_df = pickle.load(handle)