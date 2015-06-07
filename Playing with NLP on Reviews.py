
# coding: utf-8

# Play with reviews data and try clustering reviews

# In[1]:

import pandas as pd, numpy as np, os, nltk, langdetect, datetime as dt
import joblib as jb
import multiprocessing

from nltk.corpus import stopwords
stopwords = stopwords.words('english') # replace the stop word list

import string
punctuations = set(string.punctuation)


# In[2]:

print stopwords


# Import business id to yelp id crosswalk

# In[2]:

id_map = pd.read_csv("data/restaurant_ids_to_yelp_ids.csv")
id_dict = {}

# each Yelp ID may correspond to up to 4 Boston IDs
for i, row in id_map.iterrows():
    # get the Boston ID
    boston_id = row["restaurant_id"]
    
    # get the non-null Yelp IDs
    non_null_mask = ~pd.isnull(row.ix[1:])
    yelp_ids = row[1:][non_null_mask].values
    
    for yelp_id in yelp_ids:
        id_dict[yelp_id] = boston_id


# Read in yelp reviews

# In[3]:

with open("data/yelp_academic_dataset_review.json", 'r') as review_file:
    # the file is not actually valid json since each line is an individual
    # dict -- we will add brackets on the very beginning and ending in order
    # to make this an array of dicts and join the array entries with commas
    review_json = '[' + ','.join(review_file.readlines()) + ']'

# read in the json as a DataFrame
reviews = pd.read_json(review_json)

# drop columns that we won't use
reviews.drop(['review_id', 'type', 'user_id', 'votes'], 
             inplace=True, 
             axis=1)

# replace yelp business_id with boston restaurant_id
map_to_boston_ids = lambda yelp_id: id_dict[yelp_id] if yelp_id in id_dict else np.nan
reviews.business_id = reviews.business_id.map(map_to_boston_ids)

# rename first column to restaurant_id so we can join with boston data
reviews.columns = ["restaurant_id", "date", "stars", "text"]    

# drop restaurants not found in boston data
reviews = reviews[pd.notnull(reviews.restaurant_id)]

reviews.head()


# Use langdetect to detect languages. The apply below is slow. Played around with multiprocessing and it has about 2x the speed over 10000k records. Couldnt implement on iPython Notebook though. Works when running on terminal. This step takes 42 minutes on Francos laptop

# In[4]:


def detect(string):
    try:
        return langdetect.detect(string)
    except:
        return "langdetect failed"

start = dt.datetime.now()
col = reviews['text']
col2 = col.apply(detect)
print "Duration: ", dt.datetime.now() - start


# Store results

# In[5]:

reviews['language'] = col2
reviews.to_pickle('data/intermediate/langdetect.pkl')


# Present the text that langdetect had an error with:

# In[82]:

reviews = pd.read_pickle('data/intermediate/langdetect.pkl')
reviews.loc[reviews['language'] == "langdetect failed","text"]


# Also present frequency of languages:

# In[83]:

by = reviews.groupby(['language'])
by['language'].agg(len)


# Present the text for languages that are not english.

# In[84]:

for i, row in (reviews.loc[reviews['language'] != 'en',['language','text']]).iterrows():
    print row['language'], ":", row['text']


# Looks like the shorter strings are being mis-categorized. Assume these are all english.

# In[85]:

start = dt.datetime.now()
def wordlength (string):
    return len(nltk.word_tokenize(string))
reviews['length'] = reviews['text'].apply(wordlength)
print "Duration: ", dt.datetime.now() - start


# In[86]:

reviews.loc[(reviews['length'] < 10) & (reviews['language'].isin(['zh-tw','zh-cn','ja','ko']) == False ), 'language'] = 'en'
by = reviews.groupby(['language'])
by['language'].agg(len)


# we use goslate to ping google translate for translation

# In[87]:

import goslate
import time
gs = goslate.Goslate()
gs.translate("Bon pratique, prix corrects, terrasse. Bon pour un d\xe9jeuner rapide ou \xe0 toute heure.\n\nMais on n'est pas chez Fairmont Bagel, rien \xe0 voir en qualit\xe9.",'en')


# In[88]:

translate_list = []
reviews['translate'] = ''

for i in reviews.loc[reviews['language'] != 'en','text']: # where language is not english, take the text,
    translate_list.append([detect(i), gs.detect(i), i]) # prep array for comparison
    reviews.loc[reviews['text'] == i,'translate'] = gs.translate(i,'en') # translate
    time.sleep(1) # we dont want to ping too quickly


# In[ ]:

translate_list


# In[ ]:

reviews.loc[reviews['translate'] != "", ['text','translate']].head(10)


# In[2]:

reviews.to_pickle('data/intermediate/translate.pkl') # pickle results in case we need to revert back to them


# Combine translated reviews and other text

# In[8]:

reviews = pd.read_pickle('data/intermediate/translate.pkl')
reviews.loc[reviews['translate'] != '','final_text'] = reviews['translate'] # if translate is not missing, then use translate
reviews.loc[reviews['translate'] == '','final_text'] = reviews['text'] # if translate is missing, then use original text

# some assertions
assert((max(reviews['final_text'] == '') == False) == True) # no missing reviews
assert((max(reviews['final_text'].isnull()) == False) == True) # no NaN reviews


# Process Text

# In[9]:

# make same capitalization
reviews['final_text']=reviews['final_text'].str.lower()
reviews.to_pickle('data/intermediate/pre_process.pkl') # pickle results in case we need to revert back to them


# In[3]:

reviews = pd.read_pickle('data/intermediate/pre_process.pkl')


# In[6]:

# remove punctuation, stopwords, lemmatize
# make ngrams (potentially want to experiment with multiple ngrams (up to quadgrams?))

import nltk
from nltk.util import ngrams

wnl = nltk.WordNetLemmatizer()

def ngramfunc(string,gram_count):
    
    # remove punctuations
    string = ''.join(ch for ch in string if ch not in punctuations) 
    
    # not fancy list comprehension here to avoid lemmatizing twice or running over the list twice
    string_array = []
    for word in string.split():
        lemmatized = wnl.lemmatize(word) # lemmatize the word
        if lemmatized not in stopwords:
            string_array.append(lemmatized) # if lemmatized word not in stop words, add to string array
        else:
            pass
    
    # place into ngrams
    gram_array = []
    for i in ngrams(string_array,gram_count):
        gram_array.append(i)
    return gram_array


# In[5]:

# testing
reviews['final_text'][:5].apply(lambda string,gramcount: ngramfunc(string,gramcount), args=(2,))


# We write a script that can be called from a batch file to run this asynchronously

# In[6]:

script = '''
import pandas as pd
import sys
import nltk
from nltk.util import ngrams
import string
punctuations = set(string.punctuation)
from nltk.corpus import stopwords
stopwords = stopwords.words('english') # replace the stop word list

ngramcount = sys.argv[1]
varname = 'ngrams' + ngramcount
wnl = nltk.WordNetLemmatizer()

def ngramfunc(string,gram_count):
    
    # remove punctuations
    string = ''.join(ch for ch in string if ch not in punctuations) 
    
    # not fancy list comprehension here to avoid lemmatizing twice or running over the list twice
    string_array = []
    for word in string.split():
        lemmatized = wnl.lemmatize(word) # lemmatize the word
        if lemmatized not in stopwords:
            string_array.append(lemmatized) # if lemmatized word not in stop words, add to string array
        else:
            pass
    
    # place into ngrams
    gram_array = []
    for i in ngrams(string_array,gram_count):
        gram_array.append(i)
    return gram_array
    
df = pd.read_pickle('data/intermediate/pre_process.pkl')
df = df.loc[:,['restaurant_id','date','final_text']]
print "running ngrams ",ngramcount
df[varname] = df['final_text'].apply(lambda string,gramcount: ngramfunc(string,gramcount), args=(int(ngramcount),))
df.to_pickle('data/intermediate/ngrams' + ngramcount + '.pkl')
'''
filepath = open("ngram.py",'w')
filepath.writelines(script)
filepath.close()


# In[ ]:

bat = '''
start python ngram.py 1
start python ngram.py 2
start python ngram.py 3
start python ngram.py 4
'''
filepath = open("exec.bat",'w')
filepath.writelines(bat)
filepath.close()


# In[ ]:

subprocess.call('exec.bat')
def watcher():
    flag = 0
    while True:
        arraylist = os.listdir('data/intermediate')
        if ('ngrams1.pkl' in arraylist) & ('ngrams2.pkl' in arraylist) & ('ngrams3.pkl' in arraylist) & ('ngrams4.pkl' in arraylist):
            return
watcher()
print "files done"
print "deleting files"
os.remove("exec.bat")
os.remove("ngram.py")


# In[ ]:



