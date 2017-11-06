from sklearn.cluster import KMeans
import time
from gensim.models import Word2Vec
import os
import numpy as np
import pickle
import json
import marshal

os.chdir('C:\\Users\\Dell\\Desktop\\data')
model = Word2Vec.load("300features_40minwords_10context")

start = time.time() # Start time

def create_bag_of_centroids( wordlist, word_centroid_map ):

    num_centroids = max( word_centroid_map.values() ) + 1

    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    return bag_of_centroids

if __name__ == '__main__':

    model = Word2Vec.load("300features_40minwords_10context")




    # Liczba klastrow to 1/5 slow
    # average of 5 words per cluster
    word_vectors = model.wv.syn0
    num_clusters = int(word_vectors.shape[0] / 5)

    print ("Running K means")
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )


    pkl_name = "cluster_model.pkl"
    with open(pkl_name,'wb') as file:
        marshal.dump(idx, file)


    with open(pkl_name,'rb') as file:
        idx2 = marshal.load(file)

    word_centroid_map = dict(zip( model.wv.index2word, idx2 ))

    for cluster in range(0,10):

        print ("\nCluster %d" % cluster)
        #
        # Drukuje numery dla pierwszych 20 klastrow
        words = []
        for i in range(0,len(list(word_centroid_map.values()))):
            if( list(word_centroid_map.values())[i] == cluster ):
                words.append(list(word_centroid_map.keys())[i])
        print (words)