# %%
from playlists import get_features, cut_songs_modified, spotify_conn
from pathlib import Path
import pandas as pd
import json
#%%
def cut_songs_dict(tracks: dict):
    request_tracks = dict()
    while len(tracks) > 0 and len(request_tracks) < 100:
        uri, features = tracks.popitem()
        request_tracks[uri] = features
    return (tracks, request_tracks)
# %%
file_dir = Path(__file__).parent
data_dir = file_dir / 'data'
conn = spotify_conn(file_dir / 'keys.json')
path = r'c:\Users\Raleigh\Desktop\Work\song_prediction\data\mpd.slice.0-999.json'
# %%
data = json.load(open(path))
field_names = ["artist_name", "track_name"]
file_tracks = dict()
track_uri_dict = dict()
for ind, playlist in enumerate(data['playlists']):
    for track in playlist["tracks"]:
        track_uri_dict[track["track_uri"]] = [track["artist_name"], track["track_name"]]
counter = 0
while(len(track_uri_dict) > 0):
    counter +=1
    track_uri_dict, request_tracks = cut_songs_dict(track_uri_dict)
    track_uris = request_tracks.keys()
    features_res = get_features(conn, track_uris)
    for ind, val in enumerate(features_res):
        request_tracks[val['uri']] = request_tracks[val['uri']] + list(val.values())
    file_tracks.update(request_tracks)
field_names += val.keys()

df = pd.DataFrame.from_dict(data=file_tracks, orient='index', columns=field_names)
