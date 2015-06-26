from numpy import argsort
import pandas as pd
import datetime
import time
import calendar
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import csv

# a simple way to create a "document" for an inspection is to
# concatenate all the reviews that happened before the inspection date
def flatten_reviews(label_df, reviews):
    """ label_df: inspection dataframe with date, restaurant_id
        reviews: dataframe of reviews

        Returns all of the text of reviews previous to each
        inspection listed in label_df.
    """
    reviews_dictionary = {}

    n = len(label_df)

    for i, (pid, row) in enumerate(label_df.iterrows()):
        # we want to only get reviews for this restaurant that occurred before the inspection
        pre_inspection_mask = (reviews.date < row.date) & (reviews.restaurant_id == row.restaurant_id)

        # pre-inspection reviews
        pre_inspection_reviews = reviews[pre_inspection_mask]

        # join the text
        all_text = ' '.join(pre_inspection_reviews.text)

        # store in dictionary
        reviews_dictionary[pid] = all_text

        if i % 2500 == 0:
            print('{} out of {}'.format(i, n))

    # return series in same order as the original data frame
    return reviews_dictionary


def build_restaurant_id_map(csvfile):
    """ Build a map between Boston ID and Yelp ID
        :param csvfile: A CSV file containing Boston-to-Yelp ID mappings
        :return a dict containing a mapping between Boston ID and Yelp ID
    """
    id_dict = {}
    with open(csvfile, 'r') as f:
        linereader = csv.reader(f)
        for row in linereader:
            for e in row[1:]:
                if e != '':
                    id_dict[e] = row[0]
    return id_dict

def hour_to_seconds(hour):
    if hour is None or hour != hour:
        return -1
    x = time.strptime(hour, '%H:%M')
    return datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()

def date_to_seconds(text):
    if text is None or text != text:
        return -1
    x = time.strptime(text, '%Y-%m-%d')
    return calendar.timegm(x)

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # TODO something smarter

def pos_and_lemmatize(text, lemmatizer):
    words = [w.encode('ascii', 'ignore').decode('utf-8') for w in word_tokenize(text)]
    pos = nltk.pos_tag(words)
    lemmas = [lemmatizer.lemmatize(w, pos=get_wordnet_pos(p)) for (w, p) in pos]
    return lemmas
