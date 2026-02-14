from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

API_KEY = "8265bd1679663a7ea12ac168da84d2e8" # Example key, replace with your own if needed

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return send_file('index.html')

# --- Load and prepare data ---
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].dropna()

def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
new_df['title_norm'] = new_df['title'].str.lower().str.replace(" ", "")

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

def fetch_poster(movie_id, title):
    try:
        url = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US".format(movie_id, API_KEY)
        data = requests.get(url, timeout=5).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        # Fallback to movie name if no poster
        return "https://placehold.co/500x750?text={}".format(requests.utils.quote(title))
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
        # Fallback to movie name on error
        return "https://placehold.co/500x750?text={}".format(requests.utils.quote(title))

@app.route('/recommend', methods=['GET'])
def recommend():
    # Remove all spaces from search query
    movie = request.args.get('title', '').lower().replace(" ", "")
    
    if movie not in new_df['title_norm'].values:
        return jsonify({'error': 'Movie not found'})
    index = new_df[new_df['title_norm'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_posters = []
    
    for i in movie_list:
        movie_id = new_df.iloc[i[0]].movie_id
        movie_title = new_df.iloc[i[0]].title
        recommended_movies.append(movie_title)
        recommended_posters.append(fetch_poster(movie_id, movie_title))
        
    return jsonify({
        'movie_names': recommended_movies,
        'movie_posters': recommended_posters
    })

if __name__ == '__main__':
    app.run(debug=True)
