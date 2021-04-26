#%%
from data import get_x_y
from pathlib import Path
from playlists import spotify_conn, get_playlists_from_file
import numpy as np
import pickle
import pandas as pd



# file_dir = Path(__file__).parent
# data_dir = file_dir / 'data'
# spotify = spotify_conn(file_dir / 'keys.json')
# #%%
# file_path = data_dir / 'mpd.slice.0-999.json'
# # array_df = get_playlists_from_file(file_path, spotify)
# # with open('mpd.slice.0-999.pkl', 'wb') as handle:
# #     pickle.dump(array_df, handle)
# with open('mpd.slice.0-999.pkl', 'rb') as handle:
#     array_df = pickle.load(handle)

# #%%
# dict_x_y = get_x_y(array_df, average=False, iterate_arrays=False)
# X = dict_x_y['x']
# y = dict_x_y['y']

# for i, playlist in enumerate(X):
#     for ind, track in enumerate(playlist):
#         playlist[ind] = np.asarray(track, dtype=np.float32)
#     X[i] = np.asarray(playlist, dtype=object)
# X = np.array(X)
with open('all_data.pkl', 'rb') as handle:
    read_array_df = pickle.load(handle)
array_df_noNA = []
for df in read_array_df:
    if not len(df) <2:
        array_df_noNA.append(df)