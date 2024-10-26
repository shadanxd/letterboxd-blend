import numpy as np
from collections import defaultdict
from letterboxdpy import movie

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

    if len(l) == 0:
        return 0  # If no common movies, compatibility is 0

    v1 = findVector(u1_reviews, l)
    v2 = findVector(u2_reviews, l)

    finalCompatibility = findCosine(v1, v2) * 100
    return finalCompatibility

def get_movie_genres(movie_id):
    """
    Extracts genres from a movie's API response
    
    Args:
        movie_id: The ID of the movie to look up
        
    Returns:
        list: A list of genres associated with the movie
    """
    try:
        movie_details = movie.Movie(movie_id)
        return movie_details.genres
    except Exception as e:
        print(f"Error fetching genres for movie {movie_id}: {str(e)}")
        return []

def create_genre_vector(user_diary):
    """
    Creates a vector of average ratings per genre for all movies a user has rated
    
    Args:
        user_diary: List of diary entries containing movie ratings
        
    Returns:
        dict: Dictionary mapping genres to their average ratings
    """
    genre_ratings = defaultdict(list)
    total_movies = 0
    
    for entry in user_diary:
        if entry['rating'] is not None:  # Only consider rated movies
            rating = rating_to_numeric[entry['rating']]
            try:
                genres = get_movie_genres(entry['movie_id'])
                if genres:  # If we successfully got genres
                    total_movies += 1
                    for genre in genres:
                        genre_ratings[genre].append(rating)
            except Exception as e:
                print(f"Error processing entry {entry['movie_id']}: {str(e)}")
                continue
    
    # Calculate average rating and number of movies for each genre
    genre_vector = {}
    for genre, ratings in genre_ratings.items():
        if ratings:
            avg_rating = np.mean(ratings)
            # Weight by how often they watch this genre
            frequency = len(ratings) / total_movies
            genre_vector[genre] = avg_rating * frequency
    
    return genre_vector

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
    """
    Calculate overall compatibility between users based on:
    1. Rating similarity for common movies
    2. Genre preferences across ALL rated movies
    
    Args:
        u1_reviews: First user's diary entries
        u2_reviews: Second user's diary entries
        
    Returns:
        float: Compatibility score between 0 and 100
    """
    # Calculate rating-based compatibility (using common movies)
    rating_compatibility = findCompatibility(u1_reviews, u2_reviews)
    
    try:
        # Calculate genre-based compatibility (using all rated movies)
        u1_genre_vector = create_genre_vector(u1_reviews)
        u2_genre_vector = create_genre_vector(u2_reviews)
        
        if not u1_genre_vector or not u2_genre_vector:
            return rating_compatibility
            
        genre_compatibility = genre_similarity(u1_genre_vector, u2_genre_vector) * 100
        
        # Combine scores (70% rating similarity, 30% genre preference similarity)
        final_compatibility = 0.7 * rating_compatibility + 0.3 * genre_compatibility
        
        return final_compatibility
    
    except Exception as e:
        print(f"Error calculating genre compatibility: {str(e)}")
        return rating_compatibility
