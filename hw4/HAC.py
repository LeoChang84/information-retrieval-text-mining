#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
from heapq import heappush, heappop


# In[17]:


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


# In[18]:


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
    return result

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


# In[19]:


import glob
import math
from collections import Counter

directory = glob.glob('IRTM/*.txt')
countList = []
for d in sorted(directory, key=get_int):
    tokens = preprocess(d)
    count = Counter(tokens)
    countList.append(count)
# print(countList)
tf_documents = []
sorted_documents = []
for i, count in enumerate(countList):
    word2id = {key: index for index, key in enumerate(sorted(get_vocabulary_dict(countList).keys()), start=1)}
    scores = {word2id[word]: tfidf(word, count, countList) for word in count}
    sorted_words = sorted(scores.items(), key=lambda x: x[0])
    save_vector_file(str(i+1), sorted_words)
    tf_documents.append(scores)


# In[20]:


c = cos_sim(tf_documents[0], tf_documents[1])
print(c, cos_sim(tf_documents[1], tf_documents[2]))


# In[21]:


def build_table_and_heap(tf_documents):
    doc_num = len(tf_documents)
    cos_heap = []
    for i in range(doc_num):
        for j in range(i+1, doc_num):
            heappush(cos_heap, (-cos_sim(tf_documents[i], tf_documents[j]), i, j))
    return cos_heap


# In[22]:


doc_num = len(tf_documents)
merge_list = [[y] for y in range(doc_num)]
cos_heap = build_table_and_heap(tf_documents)
tree_dict = {}


# In[23]:


print(doc_num)
record_cluster = [20, 13, 8]


# In[67]:


def linking(cos_heap, record_cluster, doc_num):
    count = 0
    tree_dict = {}
    merge_list = [[y] for y in range(doc_num)]
    while cos_heap:
        e = heappop(cos_heap)
#         print(e)
        cos = e[0]
        node_x = e[1]
        node_y = e[2]
        if node_y in merge_list[node_x]: continue
        else:
            count += 1
            print(len(merge_list[node_x]), len(merge_list[node_y]))
            print('{} not in merge_list[{}]: {}'.format(node_y, node_x, merge_list[node_x]))
            print('merge merge_list[{}]: {} into merge_list[{}]: {}'.format(node_y, merge_list[node_y], node_x, merge_list[node_x]))
            docs = list(set(merge_list[node_x]) | set(merge_list[node_y]))
            for doc in sorted(docs):
                merge_list[doc] = docs
            print('get merge_list[{}]: {}'.format(node_x, merge_list[node_x]))
            for i in record_cluster:
                if i == doc_num - count:
                    tree_dict[i] = [list(item) for item in set(tuple(sorted(row)) for row in merge_list)]
#                     print(i, tree_dict[i])
        if doc_num - count <= 8: break
    return tree_dict


# In[68]:


tree_dict = linking(list(cos_heap), list(record_cluster), doc_num)


# c_heap = list(cos_heap)
# len(cos_heap)


# In[69]:


# print(doc_num)
tree_dict = linking(list(cos_heap), list(record_cluster), doc_num)


# In[74]:


with open('20.txt', 'w') as f:
    for docs in tree_dict[20]:
        for doc in docs:
            f.write(str(doc+1) + '\n')
        f.write('\n')    

with open('13.txt', 'w') as f:
    for docs in tree_dict[13]:
        for doc in docs:
            f.write(str(doc+1) + '\n')
        f.write('\n')    

with open('8.txt', 'w') as f:
    for docs in tree_dict[8]:
        for doc in docs:
            f.write(str(doc+1) + '\n')
        f.write('\n')    
        
# print(len(tree_dict[20]))
# print(len(tree_dict[13]))
# print(len(tree_dict[8]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




