import os
import glob
import pandas as pd
import re
import string
import itertools
import nltk
import time
import math
from nltk.stem import PorterStemmer
stemming = PorterStemmer()
from nltk.corpus import stopwords
stops = set(stopwords.words("english")) 
#print(stops)
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import pickle
from functools import reduce
from itertools import product
from collections import defaultdict
import json
ps =nltk.PorterStemmer()
from nltk.corpus import stopwords
stops = set(stopwords.words("english")) 
#print(stops)
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


def clean_text(text):
    str1 = " "
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stops]  # remove stopwords and stemming
    return str1.join(text)

with open('inverted_index.json') as f:
    inverted = json.load(f)

words = list(inverted.keys())

documents = set()
for key,value in inverted.items():
    documents = documents.union(set(value.keys()))

#transform the tf idf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


documentvectors = {}
for document in documents:
    vectorizer = TfidfVectorizer(vocabulary=words)
    corpus = list(pd.read_csv(f'Dataset/{document}')['Snippet'].apply(clean_text))
    x = vectorizer.fit_transform(corpus)
    documentvectors[document] = x

with open('vectorspace.pickle', 'wb') as fp:
    pickle.dump(documentvectors, fp,protocol=pickle.HIGHEST_PROTOCOL)