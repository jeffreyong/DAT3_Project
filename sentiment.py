# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv

import pandas as pd
import numpy as np

with open('positivedata.csv', 'rU') as f:
    file1 = [row for row in f]
    file1 = map(lambda s: s.strip(), file1)
    file1 = filter(None, file1)
        
pos_data_df = pd.DataFrame(file1)
pos_data_df.columns = ["Reviews"]
pos_data_df.shape
pos_data_df.head()

pos_data_df.insert(0, "Sentiment", 1)

with open('negativedata.csv', 'rU') as g:
    file2 = [row for row in g]
    file2 = map(lambda s: s.strip(), file2)
    file2 = filter(None, file2)

neg_data_df = pd.DataFrame(file2)
neg_data_df.columns = ["Reviews"]
neg_data_df.insert(0, "Sentiment", 0)

train_pos_df = pos_data_df[:len(pos_data_df)*4/5]
train_neg_df = neg_data_df[:len(neg_data_df)*4/5]

train_df = train_pos_df.append(train_neg_df)

test_pos_df = pos_data_df[len(pos_data_df)*4/5:]
test_neg_df = neg_data_df[len(neg_data_df)*4/5:]

test_df = test_pos_df.append(test_neg_df)

test_df = test_df.drop("Sentiment", axis=1)

# Count the labels in train_df to verify there are only 0 and 1
train_df.Sentiment.value_counts()
np.mean([len(s.split(" ")) for s in train_df.Reviews])
 
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text) #remove all except letters
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = CountVectorizer(
        analyzer = 'word',
        tokenizer = tokenize,
        lowercase = True,
        stop_words = 'english',
        max_features = 100)

# learn and fit the model/vectorizer and transform corpus into feature vectors

corpus_all_features = vectorizer.fit_transform(train_df.Reviews.tolist() + test_df.Reviews.tolist())

# make corpus features a numpy array
corpus_all_features = corpus_all_features.toarray()

# print the features
features_corpus = vectorizer.get_feature_names()
print features_corpus

# check there are 100 features
len(features_corpus)

# find frequency of features in corpus_all_features
fdist_features = np.sum(corpus_all_features, axis=0)
for word, count in zip(features_corpus, fdist_features):
    print count, word
"How to do this frequency distribution with nltk?"

# Create the classifier
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        corpus_all_features[0:len(train_df)], 
        train_df.Sentiment,
        train_size=0.85, 
        random_state=1234)

from sklearn.linear_model import LogisticRegression

log_mod = LogisticRegression()
log_mod = log_mod.fit(X_train, y_train)

y_pred = log_mod.predict(X_test)

# find classifier precision
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# train data with all training data to use for sentiment analysis with original test set

# train classifier, all test data
log_mod = LogisticRegression()
log_mod = log_mod.fit(X=corpus_all_features[0:len(train_df)], y=train_df.Sentiment)

#  get predictions
test_pred = log_mod.predict(corpus_all_features[len(train_df):])


# print sample of predictions
import random
sample = random.sample(xrange(len(test_pred)), 15)
for review, sentiment in zip(test_df.Reviews[15:30], test_pred[15:30]):
    print sentiment, review

''' To try 
use Random Forest
use a different sparsity coefficient, > 100, < 100?
use stopwords that include previously omitted words
test if these make our model better'''





    