import numpy as np

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
