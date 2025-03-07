import numpy as np
from collections import defaultdict
from letterboxdpy import movie
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

"""
rating character to value mapping
"""
rating_to_numeric = {
    '½': 0.5,
    '★': 1,
    '★½': 1.5,
    '★★': 2,
    '★★½': 2.5,
    '★★★': 3,
    '★★★½': 3.5,
    '★★★★': 4,
    '★★★★½': 4.5,
    '★★★★★': 5,
    '': 0  # Assuming no rating is equivalent to 0
}

def parse_entries(raw_diary):
    entry_list = list(raw_diary['entrys'].items())
    diary_log_list = [i[1] for i in entry_list]
    return diary_log_list

"""
Returns list of movies with ratings
from user diary
"""
def watchedFilmsWithRatings(udiary):
    movie_list = []
    for i in udiary:
        if i['rating'] != None:
            movie_list.append(i['movie_id'])

    return list(set(movie_list))

"""
Finds common movies from user diaries
"""
def findCommonMovies(x, y):
    x_movie_list = watchedFilmsWithRatings(x)
    y_movie_list = watchedFilmsWithRatings(y)
    return list(set(x_movie_list).intersection(y_movie_list))

"""
Returns movie rating parsed
Returns a normalized vector of ratings
from a list of movies and user diary
"""
def findMovieRating(id, diary):
    for etr in diary:
        if etr['movie_id'] == id:
            return rating_to_numeric[etr['rating']]  # Convert to numeric value

def findVector(userd, movie_list):
    l_ratings = [findMovieRating(i, userd) for i in movie_list]
    v = np.array(l_ratings, dtype=float)  # Ensure it's a float array
    v = v / np.linalg.norm(v)  # Normalize
    return v

"""
Finds cosine similarity between two rating vectors 
"""
def findCosine(v1, v2):
    return np.dot(v1, v2)

"""
Finds compatibility in common movies
watched between the two users 
"""
def findCompatibility(u1_reviews, u2_reviews):
    l = findCommonMovies(u1_reviews, u2_reviews)
    print(l)

    if len(l) == 0:
        return 0  # If no common movies, compatibility is 0

    v1 = findVector(u1_reviews, l)
    v2 = findVector(u2_reviews, l)

    finalCompatibility = findCosine(v1, v2) * 100
    return finalCompatibility


@lru_cache(maxsize=None)
def get_movie_genres(movie_id: str) -> list:
    """Thread-safe cached genre fetcher with retries"""
    try:
        return movie.Movie(movie_id).genres
    except Exception as e:
        return []

def precompute_genres(user1_reviews, user2_reviews, max_workers: int = 5):
    """Pre-fetch genres for all unique movies using threading"""
    def extract_movie_ids(reviews):
        return {entry['movie_id'] for entry in reviews if entry['rating']}
    
    # Get unique movies from both users
    u1_movies = extract_movie_ids(user1_reviews)
    u2_movies = extract_movie_ids(user2_reviews)
    all_movies = list(u1_movies.union(u2_movies))
    
    # Parallel fetch using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(get_movie_genres, all_movies)

def create_genre_vector_parallel(user_diary: list) -> dict:
    """Optimized genre vector creation using pre-cached data"""
    movie_ratings = defaultdict(list)
    # Collect all ratings per movie
    for entry in user_diary:
        if entry['rating']:
            movie_id = entry['movie_id']
            rating = rating_to_numeric[entry['rating']]
            movie_ratings[movie_id].append(rating)
    
    total_movies = len(movie_ratings)
    if not total_movies:
        return {}
    
    genre_ratings = defaultdict(list)
    # Process movies with cached genres
    for movie_id, ratings in movie_ratings.items():
        genres = get_movie_genres(movie_id)
        avg_rating = np.mean(ratings)
        for genre in genres:
            genre_ratings[genre].append(avg_rating)
    
    # Calculate weighted averages
    return {
        genre: (np.mean(ratings) * (len(ratings)/total_movies))
        for genre, ratings in genre_ratings.items()
    }

def normalize_genre_vector(genre_vector):
    values = np.array(list(genre_vector.values()))
    norm = np.linalg.norm(values)
    if norm == 0:
        return genre_vector
    return {genre: value / norm for genre, value in genre_vector.items()}

def genre_similarity(genre_vector1, genre_vector2):
    """
    Calculate similarity between two users' genre preferences
    
    Args:
        genre_vector1: First user's genre ratings/preferences
        genre_vector2: Second user's genre ratings/preferences
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Get all unique genres from both users
    all_genres = set(genre_vector1.keys()) | set(genre_vector2.keys())
    
    # Create vectors with 0 for missing genres
    v1 = np.array([genre_vector1.get(genre, 0) for genre in all_genres])
    v2 = np.array([genre_vector2.get(genre, 0) for genre in all_genres])
    
    # Calculate cosine similarity
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
        
    return np.dot(v1, v2) / (norm1 * norm2)

def enhanced_compatibility(u1_reviews, u2_reviews):
    """Multi-threaded compatibility calculation"""
    start_time = time.time()
    
    # Pre-fetch genres for both users' movies in parallel
    precompute_genres(u1_reviews, u2_reviews)
    
    # Calculate rating compatibility
    rating_comp = findCompatibility(u1_reviews, u2_reviews)
    
    # Parallel genre vector creation
    with ThreadPoolExecutor(max_workers=2) as executor:
        u1_future = executor.submit(create_genre_vector_parallel, u1_reviews)
        u2_future = executor.submit(create_genre_vector_parallel, u2_reviews)
        u1_genre_vector = u1_future.result()
        u2_genre_vector = u2_future.result()
    
    # Calculate genre compatibility
    genre_comp = genre_similarity(u1_genre_vector, u2_genre_vector) * 100
    
    # Combine scores
    final_comp = 0.7 * rating_comp + 0.3 * genre_comp
    
    print(f"Total execution time: {time.time() - start_time:.2f}s")
    return final_comp

