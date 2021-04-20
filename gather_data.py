# %%
"""
    usage:
        python print.py path-mpd/
"""
# File Imports
from sorter import alphanum_key
from models import random_forest
from playlists import spotify_conn, get_playlists_from_file

# Package Imports
import sys
import json
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

file_dir = Path(__file__).parent
data_dir = file_dir / 'data'
spotify = spotify_conn(file_dir / 'keys.json')

# %%
regressor = RandomForestRegressor(warm_start=True)
for file_path in sorted(data_dir.glob('mpd.slice.*.json'), key=alphanum_key):
    print("fullpath -> ", file_path)
    # %%
    array_df = get_playlists_from_file(file_path, spotify)
    
    # setup_model(array_df)

    regressor = random_forest(regressor, array_df)

# %%
data_dir = file_dir / 'data'
filenames = os.listdir(data_dir)