#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:39:22 2017

@author: Work
"""

%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk, re
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import wordnet

with open('positivedata.csv', 'rU') as f:
    file1 = [row for row in f]
    file1 = map(lambda s: s.strip(), file1)
    file1 = filter(None, file1)
    
with open('negativedata.csv', 'rU') as g:
    file2 = [row for row in g]
    file2 = map(lambda s: s.strip(), file2)
    file2 = filter(None, file2)
  
lemmer = wordnet.WordNetLemmatizer()
def review_to_words(raw):
    # Clean punctuation
    cleantext = raw.replace('/', ' ').replace('-', ' ').replace('"', '')
    # Remove quotation marks
    review_text = re.sub(r'^"|"$', '', cleantext)
    # Remove non-letters
    letters = re.sub("[^a-zA-Z]", " ", review_text)
    # Lower case and split into individual words
    words = letters.lower().split()
    # Convert stop words to a set
    stops = set(stopwords.words("english"))
    # Remove stop words
    relevant_words = [w for w in words if not w in stops]
    # lemmatize words
    lemmatized_words = [lemmer.lemmatize(w) for w in relevant_words]
    # Join words back into one string separated by space, return result
    return( " ".join([p for p in lemmatized_words]))
    
review_to_words(file1[0])
review_to_words(file2[0])

num_pos_reviews = len(file1)
num_neg_reviews = len(file2)

print "Cleaning the set of positive reviews....\n"
# Initialize empty list to hold cleaned positive reviews
clean_pos_reviews = []
for i in xrange(1, num_pos_reviews):
    if ((i+1)%1000 == 0):
        print "Review %d of %d\n" % (i+1, num_pos_reviews)
    clean_pos_reviews.append(review_to_words(file1[i]))

print "Cleaning the set of negative reviews....\n"
# Initialize empty list to hold cleaned negative reviews
clean_neg_reviews = []
for i in xrange(1, num_neg_reviews):
    if ((i+1)%1000 == 0):
        print "Review %d of %d\n" % (i+1, num_neg_reviews)
    clean_neg_reviews.append(review_to_words(file2[i]))

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

data = pos_data_df.append(neg_data_df)

# See what is in the data
data.groupby('sentiment').describe()

data['length'] = data['reviews'].map(lambda text: len(text))
print data.head(10)

# Count the labels in train_df to verify there are only 0 and 1
data.sentiment.value_counts()
np.mean([len(s.split(" ")) for s in data.reviews])

data.length.plot(bins=50, kind='hist')
data.length.describe()
print list(data.reviews[data.length > 1740])

data.hist(column='length', by='sentiment', bins=10)

# create stemmer and tokenizer
print stopwords.words("english")

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
 
print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer
# Initialize CountVectorizer object
vectorizer = CountVectorizer(analyzer = 'word',     \
                             tokenizer = tokenize,      \
                             ngram_range = (1, 2), \
                             preprocessor = None,   \
                             stop_words = None,     \
                             max_features = 5000)

# learn and fit the model/vectorizer and transform corpus into feature vectors
corpus_all_features = vectorizer.fit_transform(data.reviews.tolist())

# make corpus features a numpy array
corpus_all_features = corpus_all_features.toarray()

# print the features
features_corpus = vectorizer.get_feature_names()
print features_corpus
len(features_corpus)

# Sum the counts of each feature word
dist = np.sum(corpus_all_features, axis=0)

# For each, print the vocabulary word and the number of times it appears
#in the training set
for tag, count in zip(features_corpus, dist):
    print count, tag


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        corpus_all_features, 
        data.sentiment,
        test_size=0.2,
        random_state=42)

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Create the Log Regression classifier

log_mod = LogisticRegression(penalty='l2', C=1.0)
log_mod = log_mod.fit(X_train, y_train)

y_log_pred = log_mod.predict(X_test)

# find classifier precision
print(classification_report(y_test, y_log_pred))
    
# Create the GaussianNB, MultinomialNB, BernoulliNB classifiers
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
y_gnb_pred = gnb.predict(X_test)
# find classifier precision
print(classification_report(y_test, y_gnb_pred))

mnb = MultinomialNB(alpha=0.5, fit_prior=True, class_prior=None)
mnb = mnb.fit(X_train, y_train)
y_mnb_pred = mnb.predict(X_test)
# find classifier precision
print(classification_report(y_test, y_mnb_pred))

bnb = BernoulliNB(alpha=0.05, binarize=0.0, fit_prior=True, class_prior=None)
bnb = bnb.fit(X_train, y_train)
y_bnb_pred = bnb.predict(X_test)
# find classifier precision
print(classification_report(y_test, y_bnb_pred))

# Stochastic Descent Classifier
sgd = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, random_state=None, learning_rate='optimal')
sgd = sgd.fit(X_train, y_train)
y_sgd_pred = sgd.predict(X_test)
# find classifier precision
print(classification_report(y_test, y_sgd_pred))

# Linear SVC classifier
linear = LinearSVC(penalty='l2', dual=True, loss='squared_hinge')
linear = linear.fit(X_train, y_train)
y_linear_pred = linear.predict(X_test)
# find classifier precision
print(classification_report(y_test, y_linear_pred))
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

forest = RandomForestClassifier(n_estimators=45, max_depth = 77)
forest = forest.fit(X_train, y_train)
y_forest_pred = forest.predict(X_test)
# find classifier precision
print(classification_report(y_test, y_forest_pred))


classifier = {'Randomforest':RandomForestClassifier,
               'DecisionTree':DecisionTreeClassifier}

x_train, x_test, y_train, y_test = train_test_split(
        corpus_all_features, 
        data.sentiment,
        train_size=0.80,
        random_state=42)

for name, model in classifier1.items():
    for i in range(1,46):
        clf=model(max_depth = i)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print "%s (%s) had an accuracy score of %0.4f"% (name,i, score)

# Fine-tune n_estimators, max_depth= None for RandomForest
for i in range(10, 51):
    forest = RandomForestClassifier(n_estimators= i)
    forest.fit(x_train, y_train)
    score = forest.score(x_test, y_test)
    print "forest with %s estimators had an accuracy score of %0.4f"% (i, score)
    
# Fine-tune max_depth, n_estimators = 45 for RandomForest
for m in range(50, 101):
    forest = RandomForestClassifier(max_depth= m, n_estimators=45)
    forest.fit(x_train, y_train)
    score = forest.score(x_test, y_test)
    print "forest of max_depth =%s had an accuracy score of %0.4f"% (m, score)
    
classifiers2 = {'Logistic': LogisticRegression(penalty='l2', C=1.0),
               'GaussianNB':GaussianNB(),
               'BernoulliNB': BernoulliNB(alpha=0.05, binarize=0.0, fit_prior=True, class_prior=None),
               'MultinomialNB': MultinomialNB(alpha=0.5, fit_prior=True, class_prior=None),
               'Stochastic GD': SGDClassifier(loss='log', penalty='l2', alpha=0.0001, random_state=None, learning_rate='optimal'),
               'RandomForest1': RandomForestClassifier(n_estimators=44, max_depth = None),
               'RandomForest2': RandomForestClassifier(max_depth = 80),
               'RandomForest3': RandomForestClassifier(n_estimators=44, max_depth = 77)}

from sklearn.metrics import roc_auc_score, roc_curve, auc

for name, clf in classifiers2.items():
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    score = roc_auc_score(y_test, y_pred)
    print "%s had an accuracy score of %0.2f"% (name, score)
    #Create ROC curve
    pred_probas = clf.predict_proba(X_test)[:,1]

    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')

    plt.show()

    

