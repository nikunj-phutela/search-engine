#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import glob
import pandas as pd
import re
import string
import itertools
import nltk
import time
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
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy import dot
from numpy.linalg import norm


# In[11]:


with open('inverted_index.json') as f:
    inverted_index = json.load(f)


# In[12]:


with open('vectorspace.pickle',"rb") as f:
    documentvectors = pickle.load(f)


# In[13]:


def clean_text(text):
    str1 = " "
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stops]  # remove stopwords and stemming
    return str1.join(text)


# In[14]:


def one_word_query_dict(word, invertedIndex):
    row_id = []
#     res = []
    result={}
    final_list = []    
    #word = clean_text(word)
    pattern = re.compile('[\W_]+')
    word = pattern.sub(' ',word)
    if word in invertedIndex.keys():
        l = [filename for filename in invertedIndex[word].keys()]
        for filename in l:
            final_list.append(set((invertedIndex[word][filename].keys())))
        
        for i in range(len(l)):
            if l[i] not in result:
                result[l[i]] = final_list[i]
            else:
                result[l[i]] = result[l[i]].union(final_list[i])
    return result
                    
        
    


# In[15]:


def free_text_query(string,invertedIndex):
    startfreetext=time.time()
    string = clean_text(string)
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ',string)
    result = {}
    for word in string.split():
        result=one_word_query_dict(word,invertedIndex)
    endfreetext=time.time()
    print("free text time: ",endfreetext-startfreetext)
    return result


# In[16]:


def phrase_query_correct(string,inverted_index):
    string = clean_text(string)
    final_dict = {}
    pattern = re.compile('[\W_]+')
    string = pattern.sub(' ',string)
    listOfDict = []
    for word in string.split():
        result = {}
        result=one_word_query_dict(word,inverted_index)
        #print(result)
        listOfDict.append(result.copy())
    #print(len(listofDict))
    common_docs = set.intersection(*map(set, listOfDict))
    #print(common_docs)
    words_list = string.split()
    final_res = {}
    for filename in common_docs:
        ts = []
        for word in string.split():
               ts.append(inverted_index[word][filename])
        
        for word_pos_dict_no in range(0,len(ts)):
            for row_number in ts[word_pos_dict_no]:
                for positions in range(0,len(ts[word_pos_dict_no][row_number])):
                    ts[word_pos_dict_no][row_number][positions] -= word_pos_dict_no
        common_rows = set.intersection(*map(set,ts))
        for row_number in common_rows:
            final_list_of_pos = []
            for word_no in range(0,len(ts)):
                final_list_of_pos.append(ts[word_no][row_number])                    
            res = list(reduce(lambda i, j: i & j, (set(x) for x in final_list_of_pos)))
            if(len(res)>0):
                if(filename not in final_res):
                    final_res[filename] = []
                final_res[filename].append(row_number)
    return final_res
            
    


# In[17]:


words = list(inverted_index.keys())
numofwords = len(words)
wordindex = {}
for i in range(numofwords):
    wordindex[words[i]] = i


# In[18]:


def query_vec(query):
    vectorizer = TfidfVectorizer(vocabulary=list(inverted_index.keys()))
    corpus = [query]
    queryVec = vectorizer.fit_transform(corpus)
    return queryVec.toarray()[0]


# In[19]:


def doc_vec(document,rowno):
    return documentvectors[document][int(rowno)].toarray()


# In[20]:




# In[22]:


def enter_query():
    print("Please input the type of Query : \n '1' for free text queries \n '2' for phrase queries ")
    query_type = input()
    print("Please enter the query")
    query = input()
    printresult(querytype,query)


# In[27]:


def search_free_text(query,final_result,k=10):
        final_result[query]={}
        final_result[query]['snippets']=[]
        final_result[query]['time']=0
        print(query)
        start=time.time()
        if(len(query) == 1):
            res=one_word_query_dict(query,inverted_index)
        else:
            res=free_text_query(query,inverted_index)
        sim={}
        query_clean = clean_text(query)
        q_vec= query_vec(query_clean)
        if type(res) == type({}):
            for document,rows in res.items():
                for row in rows:
                    d_vec=doc_vec(document,row)
                    sim[document+','+row] = float(dot(d_vec,q_vec) /(norm(d_vec) * norm(q_vec)))
        elif type(res) == type([]):
            for i in res:
                for document,rows in i.items():
                    for row in rows:
                        d_vec=doc_vec(document,row,query)
                        sim[document+','+row]=(dot(d_vec,q_vec) /(norm(d_vec) * norm(q_vec)),norm(d_vector))

        else:
             pass
        sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
        sim_sorted= sim_sorted[0:int(k)]
        #sim_sorted= sim_sorted[0:10000]
        for k, v in sim_sorted:
             if v != 0.0:
                print("Similarity value:", v)
                doc,row=k.split(",")
                print("Document Name:",doc)
                infile = f'Dataset/{doc}'
                data = pd.read_csv(infile, skiprows = int(row) , nrows=1, usecols=[6])
                #print(data.values[0][0])
                final_result[query]['snippets'].append(data.values[0][0])
                print()
        end=time.time()
        print("Execution Time: ",end-start)
        final_result[query]['time']=end-start
    


# In[28]:


def search_phrase_text(query,final_result,k):
        final_result[query]={}
        final_result[query]['snippets']=[]
        final_result[query]['time']=0
        print(query)
        start=time.time()
        res=phrase_query_correct(query,inverted_index)
        sim={}
        query_clean = clean_text(query)
        q_vec= query_vec(query_clean)
        if type(res) == type({}):
            for document,rows in res.items():
                for row in rows:
                    d_vec=doc_vec(document,row)
                    sim[document+','+row] = float(dot(d_vec,q_vec) /(norm(d_vec) * norm(q_vec)))
        elif type(res) == type([]):
            for i in res:
                for document,rows in i.items():
                    for row in rows:
                        d_vec=doc_vec(document,row,query)
                        sim[document+','+row]=(dot(d_vec,q_vec) /(norm(d_vec) * norm(q_vec)),norm(d_vector))

        else:
             pass
        sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
        sim_sorted= sim_sorted[0:int(k)]
        #sim_sorted= sim_sorted[0:10000]
        for k, v in sim_sorted:
             if v != 0.0:
                print("Similarity value:", v)
                doc,row=k.split(",")
                print('row',row)
                print("Document Name:",doc)
                infile = f'Dataset/{doc}'
                data = pd.read_csv(infile, skiprows = int(row) , nrows=1, usecols=[6])
                #print(data.values[0][0])
                final_result[query]['snippets'].append(data.values[0][0])
                print()
        end=time.time()
        print("Execution Time: ",end-start)
        final_result[query]['time']=end-start


# In[29]:


queryfile="sample_queries"
inputfile= open(queryfile,"r")
final_result={}

qtype=input("Enter type of Query: 1.Free text 2.Phrase: ")
print("Enter number of queries: ")
k=input()
if(qtype=='1'):
    for line in (inputfile.read().split('\n')):
        search_free_text(line,final_result,k)
        resultsdf=pd.DataFrame.from_dict(final_result)
    filename='RetrievedFreeText-'+str(k)+'.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(resultsdf, handle, protocol=pickle.HIGHEST_PROTOCOL)
elif(qtype=='2'):
    for line in (inputfile.read().split('\n')):
        search_phrase_text(line,final_result,k)
        resultsdf=pd.DataFrame.from_dict(final_result)
        
    filename='RetrievedPhraseText-'+str(k)+'.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(resultsdf, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print('Enter valid query type')
    


# In[ ]:




