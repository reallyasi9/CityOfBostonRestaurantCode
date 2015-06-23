#!/usr/bin/python3

import flatten_data
import functions
import pandas as pd
import numpy as np
import time
import calendar

# Set hours to seconds from midnight
def date_to_seconds(text):
    if text is None or text != text:
        return -1
    x = time.strptime(text, '%Y-%m-%d')
    return calendar.timegm(x)

def wabbit_it(df):
    ignored_columns = ['id', 'name', '*', '**', '***']
    features = df.drop(ignored_columns, axis=1)
    for n, outcome in enumerate(['*', '**', '***']):
        with open('processed_data/train' + str(n) + '.txt', 'w') as f:
            results = df[outcome]
            for irow in range(1, df.shape[0]):
                outline = str(results[irow]) + " | "
                for icol in range(1, features.shape[1]):
                    outline += features.columns[icol] + ":" + str(features.iloc[irow, icol]) + " "
                f.write(outline)


def main():
    business_data = flatten_data.flatten_business_data('data/yelp_academic_dataset_business.json')
    business_data.set_index("business_id", inplace=True)
    checkin_data = flatten_data.flatten_checkin_data('data/yelp_academic_dataset_checkin.json')
    checkin_data.set_index("business_id", inplace=True)
    features_data = pd.read_excel('processed_data/tfidf.xlsx')
    training_data = pd.read_csv('data/train_labels.csv')
    training_data.ix[:, 'date'] = training_data.ix[:, 'date'].apply(date_to_seconds).astype('int32')
    id_dict = functions.build_restaurant_id_map('data/restaurant_ids_to_yelp_ids.csv')

    # features are named with the feature alone: add a prefix
    rename_cols = {}
    for col in features_data.columns:
        rename_cols[col] = "feature." + col
    features_data.rename(columns=rename_cols, inplace=True)

    # first, join features and training data using the row number
    df = training_data.join(features_data)

    # next, join business and checkin data on business ID (the index)
    bc_df = business_data.join(checkin_data)

    # add the restaurant ID to this
    map_to_boston_ids = lambda yid: id_dict[yid] if yid in id_dict else np.nan
    bc_df['restaurant_id'] = bc_df.index.map(map_to_boston_ids)

    # drop those businesses that are not in the boston dataset
    bc_df.dropna(axis=0, subset=['restaurant_id'], inplace=True)

    # deal with the repeated business ids by duplicating the training data
    df = df.merge(bc_df, on="restaurant_id")

    # In all features, replace spaces with underscores
    rename_cols = {}
    for col in df.columns:
        rename_cols[col] = col.lower().replace(" ", "_")
    df.rename(columns=rename_cols, inplace=True)

    # TODO other things

    # output
    df.to_csv("processed_data/joined_data.csv", index=False)
    wabbit_it(df)

    return


if __name__ == "__main__":
    main()
