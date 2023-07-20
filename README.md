# Search_Engine
## Search Engine on Environmental News Dataset
Run search.py to start the search engine that supports free-text, phrase and wild card query

Enter the query in quotes for phrase query

Enter the query in x * y or x * or * x format for wild card query



### Additional Details
Run inverted_index_generator to generate the inverted index. 

inverted_index.json is the generated inverted index

Run vectorspacegenerator.py to generate the tfidf matrix for the documents

vectorspace.pickle is the generated vector space matrix for the documents

Performance_comparison file compares the result of our search engine with elasticsearch.

ElasticSearchFinal file is used to generate the relevant results

RetrievedPickleGenerator.py is used to generate the retrieved results using our search engine

Run permuterm-new.ipynb to generate permuterm index

permuterm_mapping.pickle is the generated permuterm index required for wildcard query

sample_queries.txt  contains 50 random queries

