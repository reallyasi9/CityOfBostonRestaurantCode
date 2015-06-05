#!/usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from functions import flatten_reviews

def main():
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
    map_to_boston_ids = lambda yid: id_dict[yid] if yid in id_dict else np.nan
    reviews.business_id = reviews.business_id.map(map_to_boston_ids)

    # rename first column to restaurant_id so we can join with boston data
    reviews.columns = ["restaurant_id", "date", "stars", "text"]

    # drop restaurants not found in boston data
    reviews = reviews[pd.notnull(reviews.restaurant_id)]

    train_labels = pd.read_csv("data/train_labels.csv", index_col=0)
    submission = pd.read_csv("data/SubmissionFormat.csv", index_col=0)

    train_dictionary = flatten_reviews(train_labels, reviews)
    test_dictionary = flatten_reviews(submission, reviews)

    train_text = pd.Series(train_dictionary)[train_labels.index]

    test_text = pd.Series(test_dictionary)[submission.index]

    # create a TfidfVectorizer object with english stop words
    # and a maximum of 1500 features (to ensure that we can
    # train the model in a reasonable amount of time)
    vec = TfidfVectorizer(stop_words='english',
                          max_features=1500)

    # create the TfIdf feature matrix from the raw text
    train_tfidf = vec.fit_transform(train_text)

    # get just the targets from the training labels
    train_targets = train_labels[['*', '**', '***']].astype(np.float64)

    # create a Linear regresion object
    ols = linear_model.LinearRegression()

    # fit that object on the training TfIdf matrix and target variables
    ols.fit(train_tfidf, train_targets)

    # get the names of the features
    # feature_names = np.array(vec.get_feature_names())

    # get the features that indicate we are most and least likely to see violations
    # worst_feature_sets = [get_top_features(feature_names, ols, i, 100) for i in range(3)]
    # best_feature_sets = [get_top_features(feature_names, ols, i, 100, bottom=True) for i in range(3)]

    # create the same tfidf matrix for the test set
    # so we can make predictions based on the same features
    test_tfidf = vec.transform(test_text)

    # predict the counts for the test set
    predictions = ols.predict(test_tfidf)

    # clip the predictions so they are all greater than or equal to zero
    # since we can't have negative counts of violations
    predictions = np.clip(predictions, 0, np.inf)

    # write the submission file
    new_submission = submission.copy()
    new_submission.iloc[:, -3:] = predictions.astype(int)
    new_submission.to_csv("LinearRegression.csv")

if __name__ == "__main__":
    main()
