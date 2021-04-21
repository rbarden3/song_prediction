# %%
from playlists import get_features, cut_songs_modified, spotify_conn
from pathlib import Path
import pandas as pd
import json
import time
from sorter import alphanum_key

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
# path = data_dir / 'mpd.slice.0-999.json'
# path = data_dir / 'mpd.slice.8000-8999.json'
# %%
all_tracks = dict()
time_start = time.time()
file_times = dict()
for file_path in sorted(data_dir.glob('mpd.slice.*.json'), key=alphanum_key):
    print("file path -> ", file_path)
    data = json.load(open(file_path))
    files_tracks = dict()
    for ind, playlist in enumerate(data['playlists']):
        for track in playlist["tracks"]:
            files_tracks[track["track_uri"]] = [track["artist_name"], track["track_name"]]
    while(len(files_tracks) > 0):
        files_tracks, request_tracks = cut_songs_dict(files_tracks)
        track_uris = request_tracks.keys()
        features_res = get_features(conn, track_uris)
        for ind, val in enumerate(features_res):
            request_tracks[val['uri']] = request_tracks[val['uri']] + list(val.values())
        all_tracks.update(request_tracks)
    file_times[file_path] = time.time() - time_start
    print("file completed in: " + str(file_times[file_path]))
field_names = ["artist_name", "track_name"] + list(val.keys())
df = pd.DataFrame.from_dict(data=all_tracks, orient='index', columns=field_names)
print("results compiled in: " + str(sum(file_times.values())))
print("Average Time: " + str(sum(file_times.values())/len(file_times)))
df.to_pickle('tracks_df.pkl')
