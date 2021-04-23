# %%
from playlists import get_all_tracks
from pathlib import Path
import pandas as pd
import pickle

# %%
file_dir = Path(__file__).parent
data_dir = file_dir / 'data'
# %%

all_tracks = get_all_tracks(data_dir)
df = pd.DataFrame.from_dict(data=all_tracks, orient='index', columns=["artist_name", "track_name"])

df.to_pickle('tracks_df_noFeatures.pkl')
df2 = pd.read_pickle(file_dir / 'tracks_df_noFeatures.pkl')
with open('tracks_dict_noFeatures.pkl', 'wb') as handle:
    pickle.dump(all_tracks, handle)
with open(file_dir / 'tracks_dict_noFeatures.pkl', 'rb') as handle:
    b = pickle.load(handle)