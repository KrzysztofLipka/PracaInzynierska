import spacy
nlp = spacy.load('en')
from spacy.symbols import nsubj, VERB

st = nlp("New York is in USA and Berlin was in Germany")
# Finding a verb with a subject from below — good
verbs = set()
for possible_subject in st:
    if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
        verbs.add(possible_subject.head)

print(verbs)

for np in st.noun_chunks:
    print(np.text, np.root.text, np.root.dep_, np.root.head.text)




def doc_preprocess(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


#-------------------------------------------------

import nltk
from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize
from nltk.corpus.reader import WordListCorpusReader
import os, os.path





import pandas as pd
import numpy as np



def namesSearching(sample_text):
    try:
        #print(train_text)
        for i in sent_tokenize(sample_text):

            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            #<RB adverb (very, silently).any char except new line? (aby bylo RBR RBS)>*
            #(match 0 or more repetitions)<VERBS>*<NNP> -zreczownik osobowy+
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            #chunked.draw()
            print (chunked)



    except Exception as e:
        print(str(e))


def addingCorpus():
    path = os.path.expanduser('~/nltk_data')
    if not os.path.exists(path):
        os.mkdir(path)
    print(os.path.exists(path))
    print(nltk.data.path)
    print(path in nltk.data.path)

    nltk.data.load('corpora/cookbook/cookbook.txt', format='raw')

    reader = WordListCorpusReader('/Users/Dell/nltk_data/corpora/cookbook/', ['wordlist.txt'])

    print(reader.words())

def entitySearching(text):
    sentences = nltk.sent_tokenize(text)
    token_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]
    print(pos_sentences)
    chunked_sentences = nltk.ne_chunk_sents(pos_sentences,binary=True)

    for sent in chunked_sentences:
        for chunk in sent:
            if hasattr(chunk, "label") and chunk.label() == "NNS":
                print(chunk)


#-----------------------------------------------------------------

import os
os.chdir('C:\\Users\\Dell\\Desktop\\data')
import pickle
from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")
import marshal
from collections import defaultdict


#print(model.most_similar("food"))


#print(model.most_similar("food")[1][0])

c = [i[0] for i in model.most_similar("food")]
#print(c)

pkl_name = "cluster_model.pkl"

with open(pkl_name, 'rb') as file:
    idx2 = marshal.load(file)

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number


word_centroid_map = dict(zip(model.wv.index2word, idx2))
word_centroid_map2 = list(zip(idx2,model.wv.index2word))
print(word_centroid_map2)


#klasters = [0,1,2,3,4,5]
#defdict = defaultdict(list)

'''
for i in word_centroid_map2:
    for d,k in word_centroid_map2:
        if d in klasters:
            defdict[d].append(k)
print(defdict[0])
'''
li = [i[1] for i in word_centroid_map2 if i[0]== 3]
print(li)



# Print the first ten clusters
for cluster in range(0, 20):
    #
    # Print the cluster number
    print("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in range(0, len(list(word_centroid_map.values()))):
        if (list(word_centroid_map.values())[i] == cluster):
            words.append(list(word_centroid_map.keys())[i])
    print(words)





