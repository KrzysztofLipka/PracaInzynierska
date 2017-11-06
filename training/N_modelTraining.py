import pandas as pd
import os

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

import nltk.data
import logging

os.chdir('C:\\Users\\Dell\\Desktop\\data')

'''
def loadDataFromFile(filename, header,
                     delimiter, quoting):
    data = pd.read_csv(filename, header,
                       delimiter, quoting)
    print('Items in dataset: ' + str(data.shape[0])\
          + 'Labels in dataset : ' + str(data.shape[1]))
    print(data.shape)
    return data
'''


def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    return sentences




def parsing_sentences(labeledtrain, unlabeledtrain, column_name):
    sentences = []
    for el in labeledtrain[column_name]:
        sentences += review_to_sentences(el, tokenizer)
    for el in unlabeledtrain[column_name]:
        sentences += review_to_sentences(el, tokenizer)
    return sentences








#-------------------------------------------------------
#labeled_train = loadDataFromFile(filename = "labeledTrainData.tsv",header =  0,delimiter = "\t",quoting= 3)
#test = loadDataFromFile("testData.tsv", 0, "\t", 3)
#unlabeled_train = loadDataFromFile("unlabeledTrainData.tsv", header = 0,delimiter =  "\t", quoting = 3)

train = pd.read_csv( "labeledTrainData.tsv", header=0,
 delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0,
 delimiter="\t", quoting=3 )


sentences = parsing_sentences(train, unlabeled_train,column_name="review")




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

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = str(num_features) + "features_" +str(min_word_count) + "minwords_"+ str(context)+"context"
model.save(model_name)





