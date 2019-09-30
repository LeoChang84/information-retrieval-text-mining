
# coding: utf-8

# In[11]:


import os
import glob

file_path = 'https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt'
directory = glob.glob('*.txt')
if len(directory) == 0:
    os.system('wget' + ' ' + file_path)


# In[12]:


# Tokenization and Lowercasing everything
from nltk import word_tokenize

tokens = []

with open('28.txt', 'r') as f_in:
    for line in f_in:
        line = line.strip()
        tokens.extend(word_tokenize(line))
        tokens=[token.lower() for token in tokens if token.isalpha()]
print('Tokenize: {}'.format(tokens))


# In[13]:


# Stemming using Porter’s algorithm.
from nltk.stem.porter import *

stemmer = PorterStemmer()

singles = [stemmer.stem(t) for t in tokens]
print('Porter\'s result: {}'.format(singles))


# In[19]:


# Stopword removal.
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))
results = ([s for s in singles if s not in stops])
print(results)
print(list(set(singles) - set(results)))


# In[18]:


# Save the result as a txt file.

directory = glob.glob('result.txt')

if len(directory) == 0:
    with open('result.txt', 'w+') as f_out:
        f_out.write('\n'.join([str(r) for r in results]))
print('Done!')

