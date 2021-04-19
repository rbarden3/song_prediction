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
spotify = spotify_conn(file_dir / 'keys.json')

# %%
# try:
#     path = sys.argv[1]
# except IndexError:
# from_ = 883000
# upto = 883999

# for i in range(100):
   # build_string = "mpd.slice." + str(from_) + "-" + str(upto) + ".json"
    #path = file_dir / 'data' / build_string
    #print("path -> ", path)
data_dir = file_dir / 'data'
for file_path in sorted(data_dir.glob('mpd.slice.*.json'), key=alphanum_key):
    print("fullpath -> ", file_path)
    # %%
    array_df = get_playlists_from_file(file_path, spotify)
    # %%
    # setup_model(array_df)

    # %%
    regressor = RandomForestRegressor()
    regressor = random_forest(regressor, array_df)

    # from_ += 1000
    # upto += 1000
    # if from_ > 999000:
    #     break
# %%
data_dir = file_dir / 'data'
filenames = os.listdir(data_dir)
# for filename in sorted(filenames):
#     print(filename) 

    # if filename.startswith("mpd.slice.") and filename.endswith(".json"):
# for filename in sorted(data_dir.glob('mpd.slice.*.json'), key=alphanum_key):
# files = [ filename for filename in data_dir.glob('mpd.slice.*.json') ]
# print(files)
