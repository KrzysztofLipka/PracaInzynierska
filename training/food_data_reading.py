import os
import pandas as pd

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import marshal

"""ten modul ma za zadanie stworzenie
zbioru składnikow ooraz listy wektorow slow
ze zbioru danych ingredients v1.csv
"""
os.chdir('C:\\Users\\Dell\\Desktop\\data')

ingr = pd.read_csv('ingredients v1.csv')



# usuwanie z tekstu znacznikow Html, znakow i
# normalizacja (tylko male litery)

def recipe_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)


# sentences - lista przepsow
# ingr_set - zbior skladnikow
sentences = []
ingr_set = []
for lab,row in ingr.iterrows():
    l = recipe_to_wordlist(str(row["features.value"]))
    sentences.append(l)
    for i in set(l):
        ingr_set.append(i)



print(len(sentences))
print(len(ingr_set))


#---------------------------------------------------------------

# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)


# wczytywanie modelu -  Word2Vec.load()
os.chdir('C:\\Users\\Dell\\Desktop\\data')
model_name = "recipes_model"
model.save(model_name)

# - zapisywnaie listy skladnikow
pkl_name = "food_set.pkl"
with open(pkl_name,'wb') as file:
    marshal.dump(ingr_set, file)


with open(pkl_name,'rb') as file:
    foods = marshal.load(file)



