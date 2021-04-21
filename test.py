# %%
from playlists import get_features, cut_songs, spotify_conn
from pathlib import Path
import json
# %%
file_dir = Path(__file__).parent
data_dir = file_dir / 'data'
conn = spotify_conn(file_dir / 'keys.json')
path = r'c:\Users\Raleigh\Desktop\Work\song_prediction\data\mpd.slice.0-999.json'
# %%
data = json.load(open(path))
track_uri_arr = []
playlist = data['playlists'][0]
for track in playlist["tracks"]:
    track_uri_arr.append([track["artist_name"], track["track_name"], track["track_uri"]])

track_uri_arr = cut_songs(track_uri_arr)
features_res = get_features(conn, track_uri_arr)