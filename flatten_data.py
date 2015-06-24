#!/usr/bin/python3
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer

import functions
import json
import pandas as pd
import re
from itertools import chain
from spellcheck import SpellChecker
from nltk.stem import WordNetLemmatizer


def flatten(structure, key="", path="", flattened=None):
    """
    Recursive function for flattening a dictionary.
    If the dictionary passed has the following structure:
    {
        a: 123,
        b: ["one", "two", 3.14],
        c: {
            d: False,
            e: ["three", 4]
        }
    }
    The resulting dictionary will look something like this:
    {
        a: 123,
        b.one: True,
        b.two: True,
        b.3.14: True,
        c.d: False,
        c.e.three: True,
        c.e.4: True
    }

    As a note, this is not a universal function, and will probably fail if given something like a list of dicts.
    :param structure: The object to add to the resulting dict (any type you would usually find in json).
    :param key: The key of the object to add to the resulting dict (string).
    :param path: The path to the given structure in the original dict (string).
    :param flattened: The flattened dict, for passing between recursive iterations.
    :return: a flattened version of the input dict.
    """
    if flattened is None:
        flattened = {}
    if type(structure) not in (dict, list):
        flattened[((path + ".") if path else "") + key] = structure
    elif isinstance(structure, list):
        for i, item in enumerate(structure):
            flattened[((path + ".") if path else "") + key + "." + item] = True
    else:
        for new_key, value in structure.items():
            flatten(value, new_key, ((path + ".") if path else "") + key, flattened)
    return flattened


def flatten_business_data(jsonfile):
    """ Construct a pandas 2D dataset from the businesses json

    :param jsonfile: The name of the file to parse
    :return: A 2D pandas dataset with all fields from the JSON file flattened in a standard way.
    """
    # Load json as an array of dicts
    with open(jsonfile) as jfile:
        js = '[' + ','.join(jfile.readlines()) + ']'

    jd = json.loads(js)

    # Flatten json:
    # "categories" becomes a list of indicator variables
    # "attributes" the same, except "attributes.Parking", "attributes.Ambience", and "attributes.Good For",
    #    which have their own sub-categories
    # "neighborhoods" is another list of indicator variables
    # "hours" becomes "hours.Monday.open", "hours.Monday.close", ...

    flattened_json = []
    for obj in jd:
        json_obj = flatten(obj)
        flattened_json.append(json_obj)

    # Now actually build out the dataset
    df = pd.DataFrame(flattened_json)

    # And set the business IDs
    # map_to_boston_ids = lambda yid: yelp_to_boston_ids[yid] if yid in yelp_to_boston_ids else np.nan
    # df['restaurant_id'] = df['business_id'].map(map_to_boston_ids)

    # Drop those businesses that are not in the boston set
    # df.dropna(axis=0, subset=['restaurant_id'], inplace=True)

    # Set hours to seconds from midnight
    time_cols = []
    for d in ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
        for e in ['open', 'close']:
            col = 'hours.' + d + '.' + e
            time_cols += [col]
            df.ix[:, col] = df.ix[:, col].apply(functions.hour_to_seconds).astype('int32')

        nonsense = df['hours.' + d + '.close'] <= df['hours.' + d + '.open']
        df.ix[nonsense, 'hours.' + d + '.close'] += 24 * 3600

    # Convert columns to explicit types
    col_types = {
        'category': ['attributes.Ages Allowed',
                     'attributes.Alcohol',
                     'attributes.Attire',
                     'attributes.BYOB/Corkage',
                     'attributes.Noise Level',
                     'attributes.Price Range',
                     'attributes.Wi-Fi',
                     'stars'],
        'int32': ['review_count'] + time_cols,
        'float32': ['latitude', 'longitude']
    }
    for typ, cols in col_types.items():
        for col in cols:
            df[col] = df[col].astype(typ)

    # Everything else is a boolean
    all_typed_cols = list(chain.from_iterable(col_types.values()))
    all_typed_cols += ['name', 'business_id', 'full_address']
    for col in df.columns:
        if col not in all_typed_cols:
            df[col] = df[col].astype('bool')
            df.loc[:, col] = df.loc[:, col].fillna(False)

    # Strip out the zip code, because that might be useful
    p = re.compile(r"(\d{5})(-\d{4})?")
    df['zip'] = df['full_address'].map(lambda x: p.search(x).group(1) if p.search(x) is not None else None)

    # Convert the names to non-UTF-8 text, which screws up csv
    df['name'] = df['name'].map(lambda x: x.encode('ascii', 'ignore'))

    # And get rid of useless columns
    df.drop(['type', 'state'], inplace=True, axis=1)

    return df


def flatten_checkin_data(jsonfile):
    """ Construct a pandas 2D dataset from the checkin json

    :param jsonfile: The name of the file to parse
    :return: A 2D pandas dataset with all fields from the JSON file flattened in a standard way.
    """
    # Load json as an array of dicts
    with open(jsonfile) as jfile:
        js = '[' + ','.join(jfile.readlines()) + ']'

    jd = json.loads(js)

    # Flatten json:

    flattened_json = []
    for obj in jd:
        json_obj = flatten(obj)
        flattened_json.append(json_obj)

    # Now actually build out the dataset
    df = pd.DataFrame(flattened_json)

    # Convert columns to explicit types
    not_integer_types = ['type', 'business_id']

    # Everything else is an integer
    for col in df.columns:
        if col not in not_integer_types:
            df.loc[:, col] = df.loc[:, col].fillna(0)
            # df[col] = df[col].astype('int32')

    # And get rid of useless columns
    df.drop(['type'], inplace=True, axis=1)

    return df


class LemmaTokenizer(object):
    """
    This is a class to tokenize using WordNet Lemmatizer
    for use with TFIDF Vectorizer.
    """

    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.sc = SpellChecker()

    def __call__(self, doc):
        return functions.pos_and_lemmatize(doc, self.wnl, self.sc)


def flatten_tip_data(jsonfile):
    """ Construct a pandas 2D dataset from the tip json

    :param jsonfile: The name of the file to parse
    :return: A 2D pandas dataset with all fields from the JSON file flattened in a standard way.
    """
    # Load json as an array of dicts
    with open(jsonfile) as jfile:
        js = '[' + ','.join(jfile.readlines()) + ']'

    jd = json.loads(js)

    # Flatten json:

    flattened_json = []
    for obj in jd:
        json_obj = flatten(obj)
        flattened_json.append(json_obj)

    # Now actually build out the dataset
    df = pd.DataFrame(flattened_json)

    vectorizer = TfidfVectorizer(stop_words='english',
                                 decode_error='replace',
                                 analyzer='word',
                                 ngram_range=(1, 2),
                                 max_features=500,
                                 tokenizer=LemmaTokenizer())

    tx_data = vectorizer.fit_transform(df['text'])
    tx_data = pd.DataFrame(tx_data.todense)
    tx_data.columns = vectorizer.get_feature_names()
    tx_data.add_prefix('f.')
    df = pd.concat([df, tx_data], axis=1)

    # And get rid of useless columns
    df.drop(['type'], inplace=True, axis=1)

    return df


def main():
    # TODO Handle options for input/output
    # TODO Add flags to determine what gets read
    business_data = flatten_business_data('data/yelp_academic_dataset_business.json')
    business_data.to_csv("processed_data/business_data.csv", index=False)
    checkin_data = flatten_checkin_data('data/yelp_academic_dataset_checkin.json')
    checkin_data.to_csv("processed_data/checkin_data.csv", index=False)
    tip_data = flatten_tip_data('data/yelp_academic_dataset_tip.json')
    tip_data.to_csv("processed_data/tip_data.csv", index=False)

    # TODO continue flattening things!


if __name__ == "__main__":
    main()
