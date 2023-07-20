import os
import glob
import pandas as pd
import re
import string
import itertools
import nltk
import pickle
from nltk.stem import PorterStemmer
ps =nltk.PorterStemmer()
from nltk.corpus import stopwords
stops = set(stopwords.words("english")) 
print(stops)
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer


def get_files(files):
    doc_to_words = {}
    doc_to_words_clean = {}
    for file in files:
        complete_path = prefix_path + '/' + str(file)
        try:
            df = pd.read_csv(complete_path,sep=',')
            df.Snippet = df.Snippet.astype(str)
            text = df.Snippet
            doc_to_words[file] = text.to_list()
        except:
            continue

        
    return doc_to_words

def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stops]  # remove stopwords and stemming
    return text

def map_pos(wordList):
    i_file = {}
    for idx, word in enumerate(wordList):
        if word in i_file.keys():
            i_file[word].append(idx)
        else:
            i_file[word] = [idx]
    return i_file

def hash_pos(doc_to_words_list):
    l = []
    hash_file_word_pos = {}
    for file in doc_to_words_list.keys():
        list_word = []
        l = doc_to_words_list[file]
        for w in l:
            w = map_pos(w)
            list_word.append(w)
        hash_file_word_pos[file] = list_word
    return hash_file_word_pos

def compute_inverted_dict(hash_file_words):
    inverted_dict = {}
    for filename in hash_file_words.keys():
        for row_number in range(0,len(hash_file_words[filename])):
                for words in hash_file_words[filename][row_number]:
                    if words not in inverted_dict.keys():
                        inverted_dict[words] = {}
                    if filename not in inverted_dict[words].keys():
                        inverted_dict[words][filename] = {}
                    inverted_dict[words][filename][row_number] = hash_file_words[filename][row_number][words]
    return inverted_dict



prefix_path = '/Users/Manam/Search_Engine/Dataset'
extension = 'csv'
os.chdir(prefix_path)
filenames = glob.glob('*.{}'.format(extension))
doc_to_words = get_files(filenames)

for key,val in doc_to_words.items():
        cl_txt = [clean_text(item) for item in val]
        doc_to_words[key] = cl_txt

hash_file_words = hash_pos(doc_to_words)
inverted_dict = compute_inverted_dict(hash_file_words)
with open('inverted_index.pickle', 'wb') as handle:
    pickle.dump(inverted_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)