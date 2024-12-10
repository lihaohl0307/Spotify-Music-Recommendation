from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spotipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from spotipy.oauth2 import SpotifyClientCredentials


"""### Data Import"""

data = pd.read_csv("./spotify-dataset/data/data.csv")
songs_data = pd.read_csv('./spotify-dataset/data/data.csv')
genre_data = pd.read_csv('./spotify-dataset/data/data_by_genres.csv')
year_data = pd.read_csv('./spotify-dataset/data/data_by_year.csv')

"""### EDA"""

# visualize into decade
def get_decade(year):
    period_start = int(year/10) * 10
    decade = '{}s'.format(period_start)
    return decade

songs_data['decade'] = songs_data['year'].apply(get_decade)
year_data['decade'] = ((year_data['year'] - 1)//10)*10
song_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'key']
scaler = MinMaxScaler()
year_data[song_features] = scaler.fit_transform(year_data[song_features])


# Calculate correlation matrix for songs_data, selecting only numerical
correlation_matrix = songs_data.select_dtypes(include=np.number).corr()

# Calculate correlations and exclude the label column itself
label_column = 'popularity'
feature_correlations = songs_data.select_dtypes(include=np.number).corr()[label_column]
feature_correlations = feature_correlations.drop(label_column).sort_values(ascending=False)

"""### Cluster Songs from spotify-dataset"""
# pipeline to standardize numerical features and cluster the dataset into 20 groups using K-Means
# assign a cluster label for each song
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=20,
                                   verbose=False))
                                 ], verbose=False)
X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

"""### Connect to Spotify API"""
client_id = "7d79fe83da984659b5581054295c34c9"
client_secret = "432302ce5805441dac55670e8a6144bd"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

"""### K-means clustering on songs"""
# Reload data in seperate data frame
songs_df = pd.read_csv('./spotify-dataset/data/data.csv')

selected_features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                     'liveness', 'loudness', 'speechiness', 'valence', 'tempo',
                     'duration_ms', 'key', 'mode']

"""### PCA for Dimensionality Reduction (retain 95% variance)"""
# Standardize the Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(songs_df[selected_features])

pca = PCA(n_components=0.95, random_state=42) # 95%
pca_features = pca.fit_transform(scaled_features)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Calculate Cumulative Variance for Full Components
pca_full = PCA(random_state=42)
pca_full.fit(scaled_features)
explained_variance_full = pca_full.explained_variance_ratio_
cumulative_variance_full = np.cumsum(explained_variance_full)

# Principal Component Loadings - detailed weighted features for each PC
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'Principal Component {i+1}' for i in range(pca.n_components_)],
    index=selected_features
)

# PC and explained variance display
components_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
    'Explained Variance': explained_variance,
    'Cumulative Variance': cumulative_variance
})

"""### Elbow Method to determine optimal k for clustering"""
inertias = []
k_values = range(5, 30)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_features)
    inertias.append(kmeans.inertia_)

"""### Apply K-means Clustering"""
optimal_k = 8
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(pca_features)

# Assign clusters to the original DataFrame
songs_df['cluster'] = kmeans.labels_

# Function to retrieve song data from Spotify API
def find_song_from_spotify_OG(name, year):
    song_data = defaultdict()
    results = sp.search(q=f'track:{name} year:{year}', limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    # Basic metadata
    song_data['name'] = results['name']
    song_data['artists'] = ', '.join([artist['name'] for artist in results['artists']])
    song_data['year'] = year
    song_data['duration_ms'] = results['duration_ms']
    song_data['explicit'] = int(results['explicit'])

    # Audio features
    for key, value in audio_features.items():
        if key in selected_features:
            song_data[key] = value

    return song_data

def find_song_from_spotify(name, year):
    song_data = {}
    try:
        # Search for the track using Spotipy
        results = sp.search(q=f'track:{name} year:{year}', type='track', limit=1)

        if not results['tracks']['items']:
            print(f"No results found for '{name}' ({year}).")
            return None

        # Extract track information
        track = results['tracks']['items'][0]
        track_id = track['id']
        audio_features = sp.audio_features(track_id)

        if not audio_features or audio_features[0] is None:
            print(f"Audio features not available for '{name}' ({year}).")
            return None

        audio_features = audio_features[0]

        # Basic metadata
        song_data['name'] = track['name']
        song_data['artists'] = ', '.join([artist['name'] for artist in track['artists']])
        song_data['year'] = year
        song_data['duration_ms'] = track['duration_ms']
        song_data['explicit'] = int(track['explicit'])

        # Audio features
        for key in selected_features:
            if key in audio_features:
                song_data[key] = audio_features[key]

    except Exception as e:
        print(f"Error retrieving song data for '{name}' ({year}): {e}")
        return None

    return song_data

def recommend_songs_kmeans_with_api(input_songs, songs_df, pca_model, scaler, kmeans_model, n_recommendations=10):
    # Ensure 'cluster' column exists
    if 'cluster' not in songs_df.columns:
        raise KeyError("'cluster' column not found in songs_df. Ensure k-means clustering is performed.")

    clusters = []
    input_song_names = []
    new_songs = []  # Store newly added songs for dynamic clustering

    # for each song, find matching records from the dataset
    for song in input_songs:
        song_name = song['name']
        year = song['year']
        # print(song_name, year)
        matched_songs = songs_df[songs_df['name'].str.lower() == song_name.lower()]
        if matched_songs.empty:
            print(f"Song '{song_name}' not found in the dataset. Fetching from Spotify...")
            # Attempt to fetch song from Spotify API
            spotify_song_data = find_song_from_spotify(song_name, year)
            if spotify_song_data:
                new_songs.append(spotify_song_data)
                print(f"Added '{spotify_song_data['name']}' from Spotify.")
            else:
                print(f"Could not find '{song_name}' on Spotify.")
            continue

        input_song_names.append(song_name.lower())
        song_clusters = matched_songs['cluster'].unique()
        clusters.extend(song_clusters)

    # Dynamically cluster new songs (if any)
    if new_songs:
        new_songs_df = pd.DataFrame(new_songs)
        scaled_new_features = scaler.transform(new_songs_df[selected_features])
        pca_new_features = pca_model.transform(scaled_new_features)
        new_clusters = kmeans_model.predict(pca_new_features)
        new_songs_df['cluster'] = new_clusters
        songs_df = pd.concat([songs_df, new_songs_df], ignore_index=True)
        clusters.extend(new_clusters)

    if not clusters:
        print("No matching songs found in the dataset or Spotify.")
        return []

    # Unique clusters
    clusters = list(set(clusters))

    # Songs in the same cluster(s)
    recommended_songs = songs_df[songs_df['cluster'].isin(clusters)]

    # Exclude input songs
    recommended_songs = recommended_songs[~recommended_songs['name'].str.lower().isin(input_song_names)]

    # Randomly select recommendations
    recommended_songs = recommended_songs.sample(n=n_recommendations, random_state=42)
    return recommended_songs[['name', 'year', 'artists', 'cluster']]


"""### Evaluate popularity"""

# calculates the average popularity of songs using Spotify's Web API.
def get_average_popularity_with_spotipy(recommendations_df):
    total_popularity = 0
    song_count = 0

    for _, row in recommendations_df.iterrows():
        song_name = row['name']
        year = row['year']

        # Construct a search query with name and year
        query = f"track:{song_name} year:{year}"

        try:
            results = sp.search(q=query, type='track', limit=10)  # Fetch multiple results to filter by year
            if results and results['tracks']['items']:
                # Filter results by release year
                filtered_tracks = [
                    track for track in results['tracks']['items']
                    if track['album']['release_date'].startswith(str(year))
                ]
                if filtered_tracks:
                    # Use the first matching track
                    track = filtered_tracks[0]
                    # print(song_name, track['popularity'])
                    total_popularity += track['popularity']
                    song_count += 1
                else:
                    print(f"No exact year match found for '{song_name}' in {year}")
            else:
                print(f"No match found for '{song_name}' in {year}")
        except Exception as e:
            print(f"Error fetching data for '{song_name}' in {year}: {e}")

    if song_count == 0:
        return 0  # Avoid division by zero

    # Calculate the average popularity
    return total_popularity / song_count


"""## Enhanced K-Means Model with Weighted Genre"""
# Define feature columns
number_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
               'liveness', 'loudness', 'speechiness', 'valence', 'tempo',
               'duration_ms', 'key', 'mode']

def classify_songs_by_genre(songs_df, genre_df, feature_columns):
    """
    Assign each song in the songs_df to the closest genre based on shared features in genre_df.
    """
    # Ensure all features are numeric and explicitly cast to float64
    songs_df[feature_columns] = songs_df[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(songs_df[feature_columns].mean())
    genre_df[feature_columns] = genre_df[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(genre_df[feature_columns].mean())

    # Extract genre features and explicitly cast to float64
    genre_features = genre_df[feature_columns].values.astype(np.float64)
    classified_genres = []

    for _, song in songs_df.iterrows():
        # Extract features for the current song and explicitly cast to float64
        song_features = song[feature_columns].values.reshape(1, -1).astype(np.float64)

        # Compute distances between the song and all genres
        distances = cdist(song_features, genre_features, metric='euclidean')

        # Assign the closest genre
        closest_genre_index = distances.argmin()
        closest_genre = genre_df.iloc[closest_genre_index]['genres']
        classified_genres.append(closest_genre)

    # Add the classified genres to the songs DataFrame
    songs_df['classified_genre'] = classified_genres
    return songs_df

# Classify songs into different genres
songs_with_genre_df = classify_songs_by_genre(songs_data, genre_data, number_cols)

songs_with_genre_df.head()

# Encode the 'classified_genre' column into numeric values
label_encoder = LabelEncoder()
songs_with_genre_df['classified_genre_encoded'] = label_encoder.fit_transform(songs_with_genre_df['classified_genre'])

# Update the feature list
selected_features_with_genre = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                                'liveness', 'loudness', 'speechiness', 'valence', 'tempo',
                                'duration_ms', 'key', 'mode', 'classified_genre_encoded'] # use encoded classfied_genre(convert string to float)

# Standardize the Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(songs_with_genre_df[selected_features_with_genre])

# Apply PCA
pca_genre = PCA(n_components=0.95, random_state=42)  # 95% variance threshold
pca_features_with_genre = pca_genre.fit_transform(scaled_features)

# Calculate Explained Variance
explained_variance_genre_enhanced = pca_genre.explained_variance_ratio_
cumulative_variance_genre_enhanced = np.cumsum(explained_variance_genre_enhanced)

# apply k-means clustering
optimal_k = 8
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(pca_features_with_genre)

# Assign clusters to the original DataFrame
songs_with_genre_df['cluster'] = kmeans.labels_
