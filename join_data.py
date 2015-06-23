#!/usr/bin/python3

import flatten_data
import functions
import pandas as pd

def main():
    business_data = flatten_data.flatten_business_data('data/yelp_academic_dataset_business.json')
    checkin_data = flatten_data.flatten_checkin_data('data/yelp_academic_dataset_checkin.json')
    features_data = pd.read_excel('processed_data/tfidf.xlsx')
    training_data = pd.read_csv('data/train_labels.csv')
    id_dict = functions.build_restaurant_id_map('data/restaurant_ids_to_yelp_ids.csv')

    # first, join features and training data using the row number
    df = training_data.join(features_data)

    # next, join business and checkin data on business ID
    bc_df = business_data.join(checkin_data, on="business_id")

    # add the restaurant ID to this
    map_to_boston_ids = lambda yid: id_dict[yid] if yid in id_dict else np.nan
    bc_df['restaurant_id'] = bc_df['business_id'].map(map_to_boston_ids)

    # drop those businesses that are not in the boston dataset
    bc_df.dropna(axis=0, subset=['restaurant_id'], inplace=True)

    # deal with the repeated business ids by duplicating the training data
    df.join(bc_df, on="restaurant_id", inplace=True)

    # TODO other things

    # output
    df.to_csv("processed_data/joined_data.csv")

    return


if __name__ == "__main__":
    main()
