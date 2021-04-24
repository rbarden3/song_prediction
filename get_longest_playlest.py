# %%
"""
    usage:
        python print.py path-mpd/
"""
# File Imports
from sorter import alphanum_key
from playlists import  get_playlists_from_file_NoFeats
# Package Imports
from pathlib import Path
from tqdm import tqdm


file_dir = Path(__file__).parent
data_dir = file_dir / 'data'

def get_longest_playlest(data_dir=data_dir):
    longest_playlest_len = 0
    for file_path in tqdm(sorted(data_dir.glob('mpd.slice.*.json'), key=alphanum_key)):
    # if True:
    #     file_path = data_dir / 'mpd.slice.0-999.json'
        file_playlists = get_playlists_from_file_NoFeats(file_path)
        for playlist in file_playlists:
            if len(playlist) > longest_playlest_len: longest_playlest_len = len(playlist)
        return longest_playlest_len
    print()

print(get_longest_playlest())
            
