
# coding: utf-8

# In[1]:

import pandas as pd, numpy as np, nltk
from collections import Counter


# In[2]:

# stick everything in a function so memory gets released after load
def func(value):
    reviews = pd.read_pickle('data/intermediate/' + value +'.pkl')
    print reviews.columns
    print reviews.head(3)
    
    # find most frequent ngrams
    grams = []
    for i in reviews[value]:
        grams += i
    print "There are ",len(grams), "ngrams"
    
    counter = Counter(grams)
    counter = pd.DataFrame(counter.items())
    print counter.sort([1],ascending=0).head(30)


# In[3]:

func('ngrams2')


# In[5]:

func('ngrams3')


# In[6]:

func('ngrams4')


# Was is being lemmatized as "wa" so it's not being identified as a stop word. Need to change sequence? Stopword removal before lemmatizing? Stopwords are all positive (i.e. was instead of wasnt, etc.) so not losing negative connotation. Negative connotation might also sit in the trigrams.

# In[7]:

import nltk
wnl = nltk.WordNetLemmatizer()
wnl.lemmatize("was")


# In[ ]:



