#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:08:06 2017

@author: Work
"""

import pandas as pd
import numpy as np
import nltk, re
from nltk.corpus import stopwords

with open('positivedata.csv', 'rU') as f:
    file1 = [row for row in f]
    file1 = map(lambda s: s.strip(), file1)
    file1 = filter(None, file1)
    
with open('negativedata.csv', 'rU') as g:
    file2 = [row for row in g]
    file2 = map(lambda s: s.strip(), file2)
    file2 = filter(None, file2)
    
def review_to_simplewords(raw):
    '''
    simpler text preprocessing
    '''
    # Clean punctuation
    cleantext = raw.replace('/', ' ').replace('-', ' ').replace('."', '.')
    # Remove quotation marks   
    cleaned_text = re.sub(r'^"|"$', '', cleantext)
    # Remove non-letters
    letters = re.sub("[^a-zA-Z]", " ", cleaned_text)
    #return( " ".join(cleaned_text))
    return letters
    
review_to_simplewords(file1[0])
review_to_simplewords(file2[0])

num_pos_reviews = len(file1)
num_neg_reviews = len(file2)

# Clean all positive reviews, status updates
print "Cleaning the set of positive reviews....\n"
# Initialize empty list to hold cleaned positive reviews
clean_pos_reviews = []
for i in xrange(1, num_pos_reviews):
    if ((i+1)%1000 == 0):
        print "Review %d of %d\n" % (i+1, num_pos_reviews)
    clean_pos_reviews.append(review_to_simplewords(file1[i]))

# Clean all negative reviews, status updates
print "Cleaning the set of negative reviews....\n"
# Initialize empty list to hold cleaned negative reviews
clean_neg_reviews = []
for i in xrange(1, num_neg_reviews):
    if ((i+1)%1000 == 0):
        print "Review %d of %d\n" % (i+1, num_neg_reviews)
    clean_neg_reviews.append(review_to_simplewords(file2[i]))

# create column headings and tag the positive reviews with sentiment 1
pos_data_df = pd.DataFrame(clean_pos_reviews)
pos_data_df.columns = ["reviews"]
pos_data_df.insert(0, "sentiment", 1)
pos_data_df.shape
pos_data_df.head()

# create column headings and tag the negative reviews with sentiment 0
neg_data_df = pd.DataFrame(clean_neg_reviews)
neg_data_df.columns = ["reviews"]
neg_data_df.insert(0, "sentiment", 0)
neg_data_df.shape
neg_data_df.head()

df = pos_data_df.append(neg_data_df)

'''First, to train Word2Vec it is better not to remove stop words because Word2Vec 
needs  the broader context of the sentence in order to produce high-quality 
word vectors. So, stop words removal is optional in the functions below. 
It also might be better not to remove numbers but nevertheless numbers were removed
in this instance.'''

def review_to_wordlist(review, remove_stopwords=False ):
    # Lower case and split words
    words = review.lower().split()
    # Optionally remove stop words (default=false)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # Return a list of words
    return(words)

'''Word2Vec expects single sentences, each sentence a list of words. The input 
format needs to be a list of lists. It is not at all straightforward 
how to split a paragraph into sentences. English sentences can end with "?", "!", """, or ".", 
among other things, we cannot rely on spacing and capitalization. 
Use NLTK's punkt tokenizer for sentence splitting to handle this type of
splitting.'''

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Split a paragraph into sentences using punkt tokenizer
    raw_sentences = tokenizer.tokenize(review.strip())
    # 2. Iterate over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    # Return the list of sentences (a list of lists)
    return sentences

sentences = []  # Start with empty list of sentences

print "Splitting sentences from dataset"
for review in df["reviews"]:
    sentences += review_to_sentences(review, tokenizer)

# Check how many sentences we have in total 
print len(sentences)
print sentences[0]
print sentences[1]
print sentences[4624]
print sentences[4630]

'''There are a number of parameter choices that affect the run time and the quality of the final model that is produced. For details, see the word2vec API documentation as well as the Google documentation. 

Architecture: Architecture options are skip-gram (default) or continuous bag of words. Skip-gram could be slightly slower but produced better results.
Training algorithm: Hierarchical softmax (default) or negative sampling. Use default.
Downsampling of frequent words: The Google documentation recommends values between .00001 and .001. Values closer 0.001 used.
Word vector dimensionality: More features result in longer runtimes, and often, but not always, result in better models. Reasonable values are in tens to hundreds; used 500.
Context / window size: How many words of context should the training algorithm take into account? More is better, up to a point. Used 15
Worker threads: Number of parallel processes to run. This is computer-specific, but between 4 and 6 should work on most systems. Just put in but dont think this was used.
Minimum word count: This helps limit the size of the vocabulary to meaningful words. Any word that does not occur at least this many times across all documents is ignored. Reasonable values are between 10 and 100. In this case,
10 was tried. With bigger dataset, might use higher value. Higher values also help limit run time.'''

# Create the Word2Vec model
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 500    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If model is not trained any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "500features_10minwords_10context"
model.save(model_name)

model.doesnt_match("man woman manager restaurant".split())

model.most_similar(["man"], topn=20)
model.most_similar(["restaurant"], topn=20)
model.most_similar("enjoy")
model.most_similar(["bad"], topn=20)
model.most_similar("awful")

# Load the model that was created 
from gensim.models import Word2Vec
model = Word2Vec.load("500features_10minwords_10context")

type(model.wv.syn0)
model.wv.syn0.shape

# Create the training and testing sets manually
train_pos_df = pos_data_df[:len(pos_data_df)*4/5]
train_neg_df = neg_data_df[:len(neg_data_df)*4/5]

train = train_pos_df.append(train_neg_df)

test_pos_df = pos_data_df[len(pos_data_df)*4/5:]
test_neg_df = neg_data_df[len(neg_data_df)*4/5:]

test= test_pos_df.append(test_neg_df)

'''Word2Vec creates clusters of semantically related words, so we can use the 
similarity of words within a cluster. Grouping vectors in this way is known as
 "vector quantization." To do this, first find the centers of the word clusters, 
 using a clustering algorithm such as K-Means. In K-Means, we need to set the 
 number of clusters; the parameter "K". Small clusters, with an average of 
 only 5 words or so per cluster, might better results than large clusters with
 many words. scikit-learn is used to perform K-Means.'''

from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an average of 5 words per cluster
word_vec = model.wv.syn0
num_clusters = word_vec.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
km_clustering = KMeans(n_clusters = num_clusters)
idx = km_clustering.fit_predict(word_vec)

# Get the end time and print how long the process took
end = time.time()
time_taken = end - start
print "Time taken for K-Means clustering: ", time_taken, "seconds."

# Create a Word dictionary by mapping vocabulary word to the cluster number                                                                                            
word_centroid_map = dict(zip(model.wv.index2word, idx))

# Find out what is in the first 15 clusters

for cluster in xrange(0,15):
    #
    # Print the cluster number  
    print "\nCluster %d" % cluster
    #
    # Print all words in a certain cluster
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if( word_centroid_map.values()[i] == cluster ):
            words.append(word_centroid_map.keys()[i])
    print words
    
'''There is a  cluster (or "centroid") assignment for each word, and we can 
define a function to convert reviews into bags-of-centroids. This works just 
like Bag of Words but uses semantically related clusters instead of individual words:
The function below will give a numpy array for each review, each with a 
number of features equal to the number of clusters. Finally, we create bags of 
centroids for our training and test set, then train classifiers and find out 
how the classifiers fare in terms of accuracy.'''
def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # Iterate the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train["reviews"].size, num_clusters), dtype="float32")

clean_train_reviews = []
for review in train["reviews"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

clean_test_reviews = []
for review in test["reviews"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros((test["reviews"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1
    
# Fit a random forest and extract predictions 
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 85, max_depth = 75)

# Fitting the forest
print "Using a random forest for train data"
forest = forest.fit(train_centroids,train["sentiment"])
y_forest_pred = forest.predict(test_centroids)

from sklearn.metrics import roc_auc_score
roc_auc_score(test["sentiment"], y_forest_pred)

from sklearn.metrics import classification_report
print(classification_report(test["sentiment"], y_forest_pred))

# Fitting a multinomialNB
print "Using a MultinomialNB for train data"
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha=0.5, fit_prior=True, class_prior=None)
mnb = mnb.fit(train_centroids,train["sentiment"])
y_mnb_pred = mnb.predict(test_centroids)

roc_auc_score(test["sentiment"], y_mnb_pred)

# find classifier precision
print(classification_report(test["sentiment"], y_mnb_pred))

'''next step would be to explore putting the word2vec model into a Recurrent
Neural Network to see if the accuracy of predictions can be improved. A bigger
dataset could also be used and parameters tweaked in the model with a bigger
dataset to see if the model fares better.'''





