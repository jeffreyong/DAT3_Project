#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:39:22 2017

@author: Work
"""

import pandas as pd

with open('positivedata.csv', 'rU') as f:
    file1 = [row for row in f]
    file1 = map(lambda s: s.strip(), file1)
    file1 = filter(None, file1)
        
pos_data_df = pd.DataFrame(file1)
pos_data_df.columns = ["reviews"]
pos_data_df.shape

print pos_data_df["reviews"][0]

example1 = pos_data_df["reviews"][0]
example2 = pos_data_df["reviews"][1]

print pos_data_df["reviews"][0]
example1 = example1[3:len(example1)]



print example1.strip()

pos_data_df["reviews"][0:3]

print example1[0]
example1.strip('"')



example3 = re.sub(r'^"|"$', '', example1)
print example1, example3
example3 = pos_data_df["reviews"][3]
example_3 = re.sub(r'^"|"$', '', example3)
example_3
example4 = '12 dozens eggs in 5 baskets'

letters_only = re.sub("[^a-zA-Z]", " ", example4)

print letters_only
file1[0]

import nltk, re
nltk.download()
from nltk.corpus import stopwords
print stopwords.words("english")

def review_to_words(raw):
    # Convert a raw review to a string of words
    # Remove quotation marks
    review_text = re.sub(r'^"|"$', '', raw)
    # Remove non-letters
    letters = re.sub("[^a-zA-Z]", " ", review_text)
    # Lower case and split into individual words
    words = letters.lower().split()
    # Convert stop words to a set
    stops = set(stopwords.words("english"))
    # Remove stop words
    relevant_words = [w for w in words if not w in stops]
    # Join words back into one string separated by space, return result
    return( " ".join(relevant_words))
    
clean_review = review_to_words(file1[0])
print clean_review

num_pos_reviews = len(file1)

# Clean all positive reviews, status updates
print "Cleaning and parsing the set of positive reviews....\n"
# Initialize empty list to hold cleaned positive reviews
clean_pos_reviews = []
for i in xrange(1, num_pos_reviews):
    if ((i+1)%1000 == 0):
        print "Review %d of %d\n" % (i+1, num_pos_reviews)
    clean_pos_reviews.append(review_to_words(file1[i]))
    

    