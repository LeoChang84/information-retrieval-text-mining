
# coding: utf-8

# In[19]:


import os
import glob

file_path = 'https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt'
directory = glob.glob('*.txt')
if len(directory) == 0:
    os.system('wget' + ' ' + file_path)


# In[35]:


# Tokenization.
from nltk import word_tokenize

with open('28.txt', 'r') as f_in:
    for line in f_in:
        line = line.strip()
        tokens.extend(word_tokenize(line))
print('Tokenize: {}'.format(tokens))


# In[36]:


# Lowercasing everything.
tokens = [t.lower() for t in tokens]
print('Lowercase: {}'.format(tokens))


# In[37]:


# Stemming using Porter’s algorithm.
from nltk.stem.porter import *

stemmer = PorterStemmer()

singles = [stemmer.stem(t) for t in tokens]
print('Porter\'s result: {}'.format(singles))


# In[40]:


# Stopword removal.
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))
results = ([s for s in singles if s not in stops])
print(result)


# In[45]:


# Save the result as a txt file.

directory = glob.glob('result.txt')

if len(directory) == 0:
    with open('result.txt', 'w+') as f_out:
        f_out.write(' '.join([str(r) for r in results]))
print('Done!')

