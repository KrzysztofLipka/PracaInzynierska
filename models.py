import nltk
from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize
from nltk.corpus.reader import WordListCorpusReader
import os, os.path
import json
from bs4 import BeautifulSoup as bs
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from gensim.models import Word2Vec
from nltk.corpus import wordnet
import spacy
nlp = spacy.load('en')
import marshal

os.chdir('C:\\Users\\Dell\\Desktop\\data')
food_set = "food_set.pkl"
with open(food_set,'rb') as file:
    foods = marshal.load(file)



def cleaning_text(sample_text):
    # funkcja czyszczaca wyslany tekst
    try:
        text = bs(sample_text).get_text()

        letters_only = re.sub("[^a-zA-Z]", " ", text)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        return (" ".join(meaningful_words))

    except Exception as e:
        print(str(e))

def ingredientSearch(sample_text):
    # funkcja wyszukajaca skladniki w tekscie
    chunks = []
    try:
        text = bs(sample_text).get_text()
        print(text)
        for i in sent_tokenize(text):
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            for index in range(len(tagged) - 1):
                if tagged[index][1] == 'CD' and (tagged[index + 1][1] == 'NNS' or tagged[index + 1][1] == 'NN'):
                    if tagged[index + 1][0] in foods:
                        #chunks.append(str(tagged[index][0]) + ' ' + str(tagged[index + 1][0]))
                        chunks.append([str(tagged[index + 1][0]) ,str(tagged[index][0])])
        return (chunks)
    except Exception as e:
        print(str(e))


# wyszukiwanie podobnych slow
def search_in_model(word,model):
    os.chdir('C:\\Users\\Dell\\PycharmProjects\\praca_inz\\data')
    model = Word2Vec.load(model)
    res = model.most_similar[word]
    return [i[0] for i in res]



# wyszukiwanie osob , miejsc itp
def entity_rec(text):
    doc = nlp(text)
    l = []
    for ent in doc.ents:
        l.append((ent.text, ent.label_))
    return l



