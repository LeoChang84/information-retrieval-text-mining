#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os

filename = 'IRTM.zip'
if not os.path.isfile(filename):
    os.system('wget https://ceiba.ntu.edu.tw/course/1d2744/hw/IRTM.zip')
os.system('7z x -PIRTM2019 ' + filename)


# In[52]:


def preprocess(filename):
    from nltk import word_tokenize
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    
    tokens = []
    with open(filename, 'r') as f_in:
        for line in f_in:
            line = line.strip()
            tokens.extend(word_tokenize(line))
            tokens = [token.lower() for token in tokens if token.isalpha()]
#     print('Tokenize: {}'.format(tokens))
    stemmer = PorterStemmer()
    singles = [stemmer.stem(t) for t in tokens]
#     print('Porter\'s result: {}'.format(singles))
    stops = set(stopwords.words('english'))
    results = ([s for s in singles if s not in stops])
    return results


# In[109]:


def get_vocabulary_dict(count_list):
    import pickle
    if not os.path.isfile('dictionary.txt'):
        df = {}
        for count in count_list:
            _list = list()
            for word in count:
                if word not in df and word not in _list:
                    df[word] = 1
                    _list.append(word)
                elif word in df and word not in _list:
                    df[word] += 1
                    _list.append(word)
        with open('dictionary.txt', 'w') as f_out:
            for i, key in enumerate(sorted(df.keys()), start=1):
                f_out.write(str(i) + ' ' + key + ' ' + str(df[key]) + '\n')
        with open('dictionary.pkl', 'wb') as pkl_out:
            pickle.dump(df, pkl_out)
    with open('dictionary.pkl', 'rb') as pkl_in:
        df = pickle.load(pkl_in)
        return df
    
def tf(word, count):
    return count[word] / sum(count.values())

def n_containing(word, count_list):
    return get_vocabulary_dict(count_list)[word]

def idf(word, count_list):
    return math.log(len(count_list) / n_containing(word, count_list))

def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)

def cos_sim(docx, docy):
    from scipy import spatial
    import numpy as np
    key_union = list(set(docx.keys()) | set(docy.keys()))
    result = 1 - spatial.distance.cosine([docx[k] if k in docx else 0 for k in key_union], [docy[k] if k in docy else 0 for k in key_union])
    print(result)

def get_int(filename):
    start, end = '/', '.'
    return int(filename[filename.find(start)+1:filename.find(end)])

def save_vector_file(filename, sorted_document):
    if not os.path.isfile('vector_files/' + filename + 'txt'):
        with open('vector_files/' + filename + '.txt', 'w') as f:
            f.write(str(len(sorted_document)) + '\n')
            f.write('t_index tf-idf' + '\n')
            for t_index, tf_idf in sorted_document:
                f.write(str(t_index) + ' ' + str(tf_idf) + '\n')


# In[110]:


import glob
import math
from collections import Counter

directory = glob.glob('IRTM/*.txt')
countList = []
for d in sorted(directory, key=get_int):
    tokens = preprocess(d)
    count = Counter(tokens)
    countList.append(count)
tf_documents = []
sorted_documents = []
for i, count in enumerate(countList):
    word2id = {key: index for index, key in enumerate(sorted(get_vocabulary_dict(countList).keys()), start=1)}
    scores = {word2id[word]: tfidf(word, count, countList) for word in count}
    sorted_words = sorted(scores.items(), key=lambda x: x[0])
    save_vector_file(str(i+1), sorted_words)
    tf_documents.append(scores)


# In[111]:


cos_sim(tf_documents[0], tf_documents[1])


# In[ ]:




