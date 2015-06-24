#!/usr/bin/python3

import flatten_data
import functions
import pandas as pd
import numpy as np

def wabbit_it(df):
    ignored_columns = ['id', 'name', '*', '**', '***', 'full_address', 'city']
    features = df.drop(ignored_columns, axis=1)
    f = [open('processed_data/train0.txt', 'w'), open('processed_data/train1.txt', 'w'), open('processed_data/train2.txt', 'w')]
    for irow in range(1, df.shape[0]):
        outline = ""
        for icol in range(1, features.shape[1]):
            if str(features.iloc[irow, icol]) != "nan":
                outline += str(features.columns[icol]).replace(" ", "_") + ":" + str(features.iloc[irow, icol]) + " "
        f[0].write(str(df.ix[irow, '*']) + " | " + outline + "\n")
        f[1].write(str(df.ix[irow, '**']) + " | " + outline + "\n")
        f[2].write(str(df.ix[irow, '***']) + " | " + outline + "\n")


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

    # TODO other things

    # output
    df.to_csv("processed_data/joined_data.csv", index=False)
    wabbit_it(df)

    return


if __name__ == "__main__":
    main()
