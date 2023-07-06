import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from Levenshtein import distance
import warnings; warnings.simplefilter('ignore')

class Recommender_Model:
    def __init__(self):
        self.cleaned_data = None
        self.cleaned_data1 = None
        self.cosine_sim = None
        self.titles = None
        self.indices = None
        self.index_movie_id = None
        self.SVD = None
        self.id_map = None
        self.preprocessing()

    def preprocessing(self):
        movie_data = pd.read_csv("Datasets/movies_metadata.csv")
        self.user_rating = pd.read_csv("Datasets/ratings_small.csv")
        vote_counts = movie_data[movie_data['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movie_data[movie_data['vote_average'].notnull()]['vote_average'].astype('int')
        average_vote_score = vote_averages.mean()
        percentile_80_cutoff = np.percentile(vote_counts,80)

        cleand_data1 = movie_data[movie_data['vote_average']>=average_vote_score]
        cleand_data1 = cleand_data1[cleand_data1['vote_count']>percentile_80_cutoff]
        
        movie_data = movie_data.drop([19730, 29503, 35587])
        links_small = pd.read_csv('Datasets/links_small.csv')
        only_subset_movies = list(links_small['tmdbId'])
        cleand_data1['id'] = cleand_data1['id'].astype('int')
        self.cleaned_data = cleand_data1[cleand_data1['id'].isin(only_subset_movies)]
        self.cleaned_data['tagline'] = self.cleaned_data['tagline'].fillna('')
        
        ### genres   
        self.cleaned_data['genres'] = self.cleaned_data['genres'].apply(literal_eval)
        self.cleaned_data['genres'] = self.cleaned_data['genres'].apply(lambda x : [i['name'] for i in x])


        stemmer = SnowballStemmer('english')
        self.cleaned_data['genres'] = self.cleaned_data['genres'].apply(lambda x: [stemmer.stem(i) for i in x])
        self.cleaned_data['genres'] = self.cleaned_data['genres'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
        
        self.cleaned_data['genres'] = self.cleaned_data['genres'].apply(lambda x : list(set(x)))
        
        # original_language  
        self.cleaned_data['original_language'].unique()
        
        credits = pd.read_csv('Datasets/credits.csv')
        keywords = pd.read_csv('Datasets/keywords.csv')
        
        self.cleaned_data = self.cleaned_data.merge(credits, on='id')
        self.cleaned_data = self.cleaned_data.merge(keywords, on='id')
        
        self.cleaned_data['keywords'] = self.cleaned_data['keywords'].apply(literal_eval)
        
        self.cleaned_data['keywords'] = self.cleaned_data['keywords'].apply(lambda x : [i['name'] for i in x])
        
        self.cleaned_data['keywords'] = self.cleaned_data['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
        self.cleaned_data['keywords'] = self.cleaned_data['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
        self.cleaned_data['keywords'] = self.cleaned_data['keywords'].apply(lambda x : list(set(x)))
        
        self.cleaned_data['cast'] = self.cleaned_data['cast'].apply(literal_eval)
        
        self.cleaned_data['crew'] = self.cleaned_data['crew'].apply(literal_eval)
        
        self.cleaned_data['top_crew'] = self.cleaned_data['cast'].apply(lambda x : [i['name'] for i in x])
        
        self.cleaned_data['top_crew'] = self.cleaned_data['top_crew'].apply(lambda x : x[:2])
        
        self.cleaned_data['director'] = self.cleaned_data['crew'].apply(get_director)
        
        imp_cols = ['tagline', 'genres' ,'original_language' ,'keywords' ,'top_crew','director']
        
        self.cleaned_data1 = self.cleaned_data[imp_cols]

        self.cleaned_data1['tagline'] = self.cleaned_data1['tagline'].apply(lambda x : [x])
        self.cleaned_data1['original_language'] = self.cleaned_data1['original_language'].apply(lambda x : [x])
        self.cleaned_data1['director'] = self.cleaned_data1['director'].apply(lambda x : [x])


        self.cleaned_data1['combine'] = self.cleaned_data1['genres'] + self.cleaned_data1['original_language'] +\
                                self.cleaned_data1['keywords'] + self.cleaned_data1['top_crew'] +\
                                self.cleaned_data1['director']
        self.cleaned_data1['combine'] = self.cleaned_data1['combine'].apply(lambda x: ' '.join(x))
        
        count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        count_matrix = count.fit_transform(self.cleaned_data1['combine'])
        
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        
        self.cleaned_data = self.cleaned_data.reset_index()
        self.titles = self.cleaned_data['title']
        self.indices = pd.Series(self.cleaned_data.index, index=self.cleaned_data['title'])
        
        self.index_movie_id = self.cleaned_data[['index','id']]
        
        reader = Reader()
        data = Dataset.load_from_df(self.user_rating[['userId', 'movieId', 'rating']], reader)
        cross_validate(NormalPredictor(), data, cv=4)
        
        self.SVD = SVD()
        trainset = data.build_full_trainset()
        self.SVD.fit(trainset)
        
        self.id_map = pd.read_csv('Datasets/links_small.csv')[['movieId', 'tmdbId']]
        self.id_map['tmdbId'] = self.id_map['tmdbId'].apply(convert_int)
        self.id_map.columns = ['movieId', 'id']
        self.id_map = self.id_map.merge(self.cleaned_data[['title', 'id']], on='id').set_index('title')
        self.indices_map = self.id_map.set_index('id')
        self.user_rating.drop(columns=['timestamp'], inplace=True)
    
    def hybrid2(self, userId, title1, title2, title3):
        idx1 = self.indices[title1]
        idx2 = self.indices[title2]
        idx3 = self.indices[title3]

        tmdbId1 = self.id_map.loc[title1]['id']
        tmdbId2 = self.id_map.loc[title2]['id']
        tmdbId3 = self.id_map.loc[title3]['id']

        movie_id1 = self.id_map.loc[title1]['movieId']
        movie_id2 = self.id_map.loc[title2]['movieId']
        movie_id3 = self.id_map.loc[title3]['movieId']

        sim_scores = list(enumerate(self.cosine_sim[int(idx1)] + self.cosine_sim[int(idx2)] + self.cosine_sim[int(idx3)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:56]
        movie_indices = [i[0] for i in sim_scores]

        movies = self.cleaned_data.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
        movies['est'] = movies['id'].apply(lambda x: self.SVD.predict(userId, self.indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)
        return movies.head(10)
    
    def get_similar_users(self, movie_names):
        movie_ids = [self.cleaned_data.loc[self.cleaned_data['title'] == movie]['id'].iloc[0] for movie in movie_names]
        new_user_ratings = pd.DataFrame({
            'userId': [max(self.user_rating['userId']) + 1] * len(movie_ids),
            'movieId': movie_ids,
            'rating': [5.0] * len(movie_ids)
        })

        merged_ratings = pd.concat([self.user_rating, new_user_ratings], ignore_index=True)

        user_item_matrix = merged_ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

        new_user_vector = user_item_matrix.loc[user_item_matrix.index[-1]].values.reshape(1, -1)
        user_similarity = cosine_similarity(user_item_matrix.values[:-1], new_user_vector)

        similar_users_indices = user_similarity.argsort(axis=0)[-10:].flatten()[::-1]

        similar_users_similarity = user_similarity[similar_users_indices].flatten()
        similar_users = user_item_matrix.iloc[similar_users_indices]

        similar_users_df = pd.DataFrame({'userId': similar_users.index, 'Similarity': similar_users_similarity})
        return similar_users_df
        
    def suggest(self, movies) -> pd.DataFrame:
        similar_id = self.get_similar_users(movies).iloc[0, 0]
        return self.hybrid2(similar_id, movies[0], movies[1], movies[2])
    
    def get_movie_info(self, movie_name: str) -> dict:
        movie_info = {}
        record = self.cleaned_data[self.cleaned_data['title'] == movie_name]
        movie_info['title'] = record['title'].to_numpy()[0]
        movie_info['overview'] = record['overview'].to_numpy()[0]
        movie_info['language'] = get_language_name(record['original_language'].to_numpy()[0])
        movie_info['genres'] = record['genres'].to_numpy()
        return movie_info
    
    def find_nearest_movie(self, movie_name: str) -> tuple:
        lowest_distance = float('inf')
        closest_movie = ''
        for movie in self.cleaned_data['title']:
            current_distance = levenshtein_distance(movie_name.replace(" ", '').lower(), movie.replace(" ", '').lower())
            if current_distance < lowest_distance:
                lowest_distance = current_distance
                closest_movie = movie
        return (closest_movie, lowest_distance)
    
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return ""

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

def levenshtein_distance(name1: str, name2: str) -> int:
    return distance(name1, name2)

def get_language_name(code:str) -> str:
    language_dict = {
        'en': 'English',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'ja': 'Japanese',
        'zh-cn': 'Chinese',
        'ru': 'Russian',
        'pt': 'Portuguese',
        'ar': 'Arabic',
        'hi': 'Hindi'
    }
    
    if code in language_dict:
        return language_dict[code]
    else:
        return 'Unknown Language'