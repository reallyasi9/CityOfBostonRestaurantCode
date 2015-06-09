#!/usr/bin/python3

import json
import pandas as pd
import datetime
import time
import numpy as np


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
        json_obj = {}
        for k, v in obj.items():
            # Collect categories
            if k == 'categories':
                for cat in v:
                    json_obj['categories.' + cat] = True

            # Collect attributes
            elif k == 'attributes':
                for attr, val in v.items():
                    if attr == 'Parking':
                        for prk, pval in val.items():
                            json_obj['attributes.Parking.' + prk] = pval
                    elif attr == 'Ambience':
                        for amb, aval in val.items():
                            json_obj['attributes.Ambience.' + amb] = aval
                    elif attr == 'Good For':
                        for gf, gval in val.items():
                            json_obj['attributes.Good For.' + gf] = gval
                    else:
                        json_obj['attributes.' + attr] = val

            # Collect business hours
            elif k == 'hours':
                for d in ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
                    for e in ['open', 'close']:
                        json_obj['hours.' + d + '.' + e] = v.get(d, {}).get(e, None)

            # Collect neighborhoods
            elif k == 'neighborhoods':
                for n in v:
                    json_obj['neighborhoods.' + n] = True

            # Collect everything else
            else:
                json_obj[k] = v
        flattened_json.append(json_obj)

    # Now actually build out the dataset
    df = pd.DataFrame(flattened_json)

    # And get rid of useless columns
    df.drop(['type', 'state', 'open'], inplace=True, axis=1)

    # And set the business IDs
    map_to_boston_ids = lambda yid: yelp_to_boston_ids[yid] if yid in yelp_to_boston_ids else np.nan
    df.business_id = df.business_id.map(map_to_boston_ids)

    # Set NaNs from certain columns to meaningful values
    for col in df.columns.values.tolist():
        if col[:13] == 'neighborhoods':
            df.ix[pd.isnull(df[col]), col] = False

    # Set hours to seconds from midnight
    def hour_to_seconds(hour):
        if hour is None:
            return None
        x = time.strptime(hour, '%H:%M')
        return datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()

    for d in ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
        for e in ['open', 'close']:
            col = 'hours.' + d + '.' + e
            df.ix[:, col] = df.ix[:, col].apply(hour_to_seconds)

        nonsense = df['hours.' + d + '.close'] <= df['hours.' + d + '.open']
        df.ix[nonsense, 'hours.' + d + '.close'] += 24 * 3600

    return df


def main():
    id_dict = build_restaurant_id_map('data/restaurant_ids_to_yelp_ids.csv')

    with open("data/yelp_academic_dataset_review.json", 'r') as review_file:
        # the file is not actually valid json since each line is an individual
        # dict -- we will add brackets on the very beginning and ending in order
        # to make this an array of dicts and join the array entries with commas
        review_json = '[' + ','.join(review_file.readlines()) + ']'

    # read in the json as a DataFrame
    reviews = pd.read_json(review_json)

    # drop columns that we won't use
    reviews.drop(['review_id', 'type'],
                 inplace=True,
                 axis=1)

    # replace yelp business_id with boston restaurant_id
    map_to_boston_ids = lambda yid: id_dict[yid] if yid in id_dict else np.nan
    reviews.business_id = reviews.business_id.map(map_to_boston_ids)

    # TODO continue flattening things!

if __name__ == "__main__":
    main()
