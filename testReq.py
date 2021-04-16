import os
import json
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Give these args to command line
file_dir = Path(__file__).parent
keys  = json.load(open(file_dir / 'keys.json'))
# Set environment variables
os.environ['SPOTIPY_CLIENT_ID'] = keys['SPOTIPY_CLIENT_ID']
os.environ['SPOTIPY_CLIENT_SECRET'] = keys['SPOTIPY_CLIENT_SECRET']


birdy_uri = 'spotify:artist:2WX2uTcsvV5OnS0inACecP'
spotify = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials())

results = spotify.artist_top_tracks(birdy_uri)
tracks = results['tracks'][0]['name']
print("results -> %s", tracks)

while results['next']:
    results = spotify.next(results)
    albums.extend(results['tracks'])

for album in albums:
    print(album['name'])
