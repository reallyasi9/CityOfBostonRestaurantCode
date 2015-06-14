#!/usr/bin/python3

import json
import pandas as pd
import datetime
import time
import numpy as np
import re
from numpy import nan


def build_restaurant_id_map(csvfile):
    """ Build a map between Boston ID and Yelp ID
        :param csvfile: A CSV file containing Boston-to-Yelp ID mappings
        :return a dict containing a mapping between Boston ID and Yelp ID
    """
    id_map = pd.read_csv(csvfile)
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

    return id_dict

def flatten(structure, key="", path="", flattened=None):
    if flattened is None:
        flattened = {}
    if type(structure) not in(dict, list):
        flattened[((path + ".") if path else "") + key] = structure
    elif isinstance(structure, list):
        for i, item in enumerate(structure):
            flattened[((path + ".") if path else "") + key + "." + item] = True
    else:
        for new_key, value in structure.items():
            flatten(value, new_key, ((path + ".") if path else "") + key, flattened)
    return flattened

def flatten_business_data(jsonfile, yelp_to_boston_ids):
    """ Construct a pandas 2D dataset from the businesses json

    :param jsonfile: The name of the file to parse
    :param yelp_to_boston_ids: A dict that maps from yelp IDs to Boston IDs
    :return: A 2D pandas dataset, keyed by Boston ID, with all fields from the JSON file flattened in a standard way.
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
    
    # TODO: Strip out the zip code, because that might be useful

    # And get rid of useless columns
    df.drop(['type', 'state', 'open', 'full_address', 'name'], inplace=True, axis=1)

    # And set the business IDs
    map_to_boston_ids = lambda yid: yelp_to_boston_ids[yid] if yid in yelp_to_boston_ids else np.nan
    df.business_id = df.business_id.map(map_to_boston_ids)

    # Set hours to seconds from midnight
    def hour_to_seconds(hour):
        if hour is None or hour != hour:
            return -1
        x = time.strptime(hour, '%H:%M')
        return datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()

    for d in ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
        for e in ['open', 'close']:
            col = 'hours.' + d + '.' + e
            df.ix[:, col] = df.ix[:, col].apply(hour_to_seconds)

        nonsense = df['hours.' + d + '.close'] <= df['hours.' + d + '.open']
        df.ix[nonsense, 'hours.' + d + '.close'] += 24 * 3600
        
    # Set NaNs from certain columns to meaningful values
    for col in df.columns.values.tolist():
        if col[:5] != 'hours.':
            df.ix[pd.isnull(df[col]), col] = False

    # Rename the column I am going to use as the index
    df.rename(columns={'business_id':'restaurant_id'}, inplace=True)
    
    # Replace empty or undefined values with NaN
    df.fillna(np.nan)
    df.replace("False", np.nan, inplace=True)
    df.replace(False, np.nan, inplace=True)
    
    return df


def main():
    id_dict = build_restaurant_id_map('data/restaurant_ids_to_yelp_ids.csv')
    business_data = flatten_business_data('data/yelp_academic_dataset_business.json', id_dict)
    business_data.to_csv("processed_data/business_data.csv", index=False)
    # TODO continue flattening things!

if __name__ == "__main__":
    main()