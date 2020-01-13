#!/usr/bin/env python
# coding: utf-8

# In[2]:


def generate_testing_set(training_set):
    from glob import glob
    
    directory = glob('IRTM/*')
    for l in training_set:
        directory.remove(l)
    return directory


# In[3]:


def get_stop_word_list():
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))


# In[4]:


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


# In[42]:


def get_vocabulary_dict(count_list):
    import pickle
    import os
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


# In[208]:


def feature_extraction(df, classified_num, word2id):
    import math
    feature_dict, feature_voc = {}, {}
    for key in word2id:
        sigma_k = len(df[df['wordIdx'] == word2id[key]])
        T_P = sigma_k / 195
        N_P = 1 - T_P
        score = 0
        for i in range(1, classified_num+1):
            k = (len(df[(df['classIdx'] == str(i)) & (df['wordIdx'] == word2id[key])]))
            E_1 = T_P * (15/195)
            E_0 = N_P * (15/195)
            if E_1 == 0 or E_0 == 0: continue
            score += ((k-E_1)**(2))/E_1 + ((15-k-E_0)**(2))/E_0
#             print(k, E_1, 15-k, E_0)
        if word2id[key] not in feature_dict:
            feature_dict[word2id[key]] = score
        else:
            if feature_dict[word2id[key]] < score: feature_dict[word2id[key]] = score
#         print(key ,score)
    #     break
    feature_voc = [k for k, v in sorted(feature_dict.items(), key=lambda item: item[1], reverse=True)]
    return feature_voc[:500]


# In[209]:


def MNB(df, feature_500_voc):
    #Using dictionaries for greater speed
    df_dict = df.to_dict()
    new_dict = {}
    prediction = []
    #new_dict = {docIdx : {wordIdx: count},....}
    for idx in range(len(df_dict['docIdx'])):
        docIdx = df_dict['docIdx'][idx]
        wordIdx = df_dict['wordIdx'][idx]
        count = df_dict['count'][idx]
        try: 
            new_dict[docIdx][wordIdx] = count 
        except:
            new_dict[df_dict['docIdx'][idx]] = {}
            new_dict[docIdx][wordIdx] = count
    #Calculating the scores for each doc
    for docIdx in new_dict:
        score_dict = {}
        #Creating a probability row for each class
        for classIdx in range(1,classified_num+1):
            classIdx = str(classIdx)
            score_dict[classIdx] = 1
            #For each word:
            for wordIdx in new_dict[docIdx]:
                if wordIdx in feature_500_voc:
                    try:
                        probability=Pr_dict[wordIdx][classIdx]         
                        score_dict[classIdx]+=np.log(probability)
                    except:
                        #Missing V will have log(1+0)*log(a/16689)=0 
                        score_dict[classIdx] += 0                        
        #Get class with max probabilty for the given docIdx 
        max_score = max(score_dict, key=score_dict.get)
        prediction.append(max_score)
        
    return prediction


# In[210]:


import os
import pandas as pd
import numpy as np
from collections import Counter


countList, df_arrays = [], []
total, classified_num = 0, 0
training_set = []

#Training label
with open('training.txt') as train_label:
    for line in train_label:
        line = line.strip().split(' ')
        tokens = []
        for i in line[1:]:
            training_set.append('IRTM/{}.txt'.format(i))
            tokens += preprocess('IRTM/{}.txt'.format(i))
            count = Counter(tokens)
            countList.append(count)
    testing_set = generate_testing_set(training_set)
    word2id = {key: index for index, key in enumerate(sorted(get_vocabulary_dict(countList).keys()), start=1)}
    train_label.seek(0, 0)
    for line in train_label:
        line = line.strip().split(' ')
        classified_num += 1
        for i in line[1:]:
            tokens = []
            tokens = preprocess('IRTM/{}.txt'.format(i))
            count = Counter(tokens)
            for key in count:
                df_array = []
                df_array.append(int(i))
                df_array.append(word2id[key])
                df_array.append(count[key])
                df_array.append(line[0])
                df_arrays.append(df_array)
    df = pd.DataFrame(df_arrays, columns=['docIdx', 'wordIdx', 'count', 'classIdx'])

feature_500_voc = feature_extraction(df, classified_num, word2id)


# In[211]:


#Alpha value for smoothing
a = 1
V = len(feature_500_voc)
#Calculate probability of each word based on class
pb_ij = df.groupby(['classIdx','wordIdx'])
pb_j = df.groupby(['classIdx'])
Pr =  (pb_ij['count'].sum() + a) / (pb_j['count'].sum() + V)
#Unstack series
Pr = Pr.unstack()

#Replace NaN or columns with 0 as word count with a/(count+|V|+1)
for c in range(1,classified_num+1):
    Pr.loc[str(c),:] = Pr.loc[str(c),:].fillna(a/(pb_j['count'].sum()[str(c)] + V))


# In[212]:


test_array = []
for line in testing_set:
    df_arrays = []
    tokens = []
    tokens = preprocess(line)
    count = Counter(tokens)
    for key in count:
        if key in word2id:
            df_array = []
            df_array.append(int(i))
            df_array.append(word2id[key])
            df_array.append(count[key])
            df_array.append(line[0])
            df_arrays.append(df_array)
    
    df = pd.DataFrame(df_arrays, columns=['docIdx', 'wordIdx', 'count', 'classIdx'])
    predict = MNB(df, feature_500_voc)
    test_array.append([line.split('/')[1].split('.')[0], predict[0]])
# print(test_array)
df = pd.DataFrame(sorted(test_array), columns=['Id','Value'])
# print(sorted(test_array))
df.to_csv('HW3.csv', index =False)


# In[ ]:





# In[ ]:




