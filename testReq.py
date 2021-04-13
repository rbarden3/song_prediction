import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Give these args to command line
export SPOTIPY_CLIENT_ID='cf066d284bdd459f9480f4d682555e48'
export SPOTIPY_CLIENT_SECRET='e7453f0113a042ada35df372168a3481'


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
