# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np

with open('positivedata.csv', 'rU') as f:
    file1 = [row for row in f]
    file1 = map(lambda s: s.strip(), file1)
    file1 = filter(None, file1)
        
pos_data_df = pd.DataFrame(file1)
pos_data_df.columns = ["reviews"]
pos_data_df.insert(0, "sentiment", 1)

pos_data_df.shape
pos_data_df.head()


with open('negativedata.csv', 'rU') as g:
    file2 = [row for row in g]
    file2 = map(lambda s: s.strip(), file2)
    file2 = filter(None, file2)

neg_data_df = pd.DataFrame(file2)
neg_data_df.columns = ["reviews"]
neg_data_df.insert(0, "sentiment", 0)

## combine the positive and negative reviews into a single dataframe
data = pos_data_df.append(neg_data_df)

# Count the labels in train_df to verify there are only 0 and 1
data.sentiment.value_counts()
np.mean([len(s.split(" ")) for s in data.reviews])
 
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
        max_features = 500)

# learn and fit the model/vectorizer and transform corpus into feature vectors

corpus_all_features = vectorizer.fit_transform(data.reviews.tolist())

# make corpus features a numpy array
corpus_all_features = corpus_all_features.toarray()

# print the features
features_corpus = vectorizer.get_feature_names()
print features_corpus

# check there are 500 features
len(features_corpus)

# find frequency of features in corpus_all_features
fdist_features = np.sum(corpus_all_features, axis=0)
for word, count in zip(features_corpus, fdist_features):
    print count, word

# Create the classifier
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        corpus_all_features, 
        data.sentiment,
        train_size=0.85, 
        random_state=1234)

from sklearn.linear_model import LogisticRegression

log_mod = LogisticRegression()
log_mod = log_mod.fit(X_train, y_train)

y_pred = log_mod.predict(X_test)

# find classifier precision
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# print sample of predictions
for review, sentiment in zip(data.reviews[15:50], y_pred[15:50]):
    print sentiment, review

''' To try 
use Random Forest
use a different sparsity coefficient, > 500, < 500?
use stopwords that include previously omitted words
test if these make our model better'''





    