#!/usr/bin/python3
import re

import functions
import pandas as pd
import numpy as np
from progress.bar import Bar


def wabbit_it(business_data, checkin_data, tip_data, review_data, training_data, df):
    return


class Closest(object):
    data = pd.DataFrame()
    cols = []
    bar = None
    # tick = 0

    def __init__(self, df, cols, max_lines):
        self.data = df
        self.cols = cols
        self.bar = Bar('Compressing Time',
                       suffix='%(percent)d%% (%(index)d/%(max)d), %(avg).3f sec/row, ETA %(eta_td)s', max=max_lines)

    def __call__(self, row):
        self.tick += 1
        if self.tick % 100 == 0:
            self.bar.next(100)
        found = self.data[(self.data.restaurant_id == row.restaurant_id) & (self.data.date <= row.date)]
        if found.shape[0] == 0:
            # FIXME Do something smarter than averaging?
            found = self.data[(self.data.restaurant_id == row.restaurant_id)][self.cols].mean()
        else:
            found = found[self.cols].sum()
        # FIXME Sometimes NaNs appear if I am missing the restaurant ID.  What to do?
        found.fillna(0, inplace=True)
        row[self.cols] = found
        return row


def create_evaluation_data(business_data, checkin_data, tip_data, review_data, tip_features, review_features,
                           target_data):
    df = target_data.copy()
    bc_df = business_data.join(checkin_data)
    closest_tip = Closest(tip_data, tip_features, df.shape[0])
    df = pd.concat([df, pd.DataFrame(np.zeros((df.shape[0], len(tip_features))), columns=tip_features)], axis=1)
    df = df.apply(closest_tip, axis=1)
    closest_review = Closest(review_data, review_features, df.shape[0])
    df = pd.concat([df, pd.DataFrame(np.zeros((df.shape[0], len(review_features))), columns=review_features)], axis=1)
    df = df.apply(closest_review, axis=1)

    df = df.join(bc_df, on='restaurant_id')
    return df


def main():
    business_data = pd.DataFrame.from_csv('processed_data/business_data.csv', index_col=None)
    checkin_data = pd.DataFrame.from_csv('processed_data/checkin_data.csv', index_col=None)
    tip_data = pd.DataFrame.from_csv('processed_data/tip_data.csv', encoding="latin_1", index_col=None)
    review_data = pd.DataFrame.from_csv('processed_data/review_data.csv', encoding="latin_1", index_col=None)
    training_data = pd.DataFrame.from_csv('data/train_labels.csv', index_col=None)

    # Convert training date to seconds for easier maths
    training_data.ix[:, 'date'] = training_data.ix[:, 'date'].apply(functions.date_to_seconds).astype('int32')

    # Figure out the names of the TFIDF feature columns
    feature_re = re.compile(r'^[tr]\.')
    tip_features = [col for col in tip_data.columns if feature_re.match(col) is not None]
    review_features = [col for col in review_data.columns if feature_re.match(col) is not None]

    # Add restaurant IDs to everything
    id_dict = functions.build_restaurant_id_map('data/restaurant_ids_to_yelp_ids.csv')
    map_to_boston_ids = lambda yid: id_dict[yid] if yid in id_dict else np.nan

    for df in [business_data, checkin_data, tip_data, review_data]:
        df['restaurant_id'] = df['business_id'].map(map_to_boston_ids)
        df.drop(['business_id'], axis=1, inplace=True)
        # drop those businesses that are not in the Boston dataset
        df.dropna(axis=0, subset=['restaurant_id'], inplace=True)

    # FIXME Do something other than average the values across matching restaurant IDs?
    # This works because the only non-numeric fields are restaurant_id and business_id.
    # restaurant_id becomes the new index after grouping, so it is also not a normal column.
    business_data = business_data.groupby('restaurant_id', sort=False).mean()
    checkin_data = checkin_data.groupby('restaurant_id', sort=False).mean()

    # Sum the TFIDF features for everything that comes before in time for a given tip or review
    # FIXME maybe don't drop userid?
    tip_data.reset_index(inplace=True)
    tip_data.drop(['user_id', 'text'], axis=1, inplace=True)
    review_data.reset_index(inplace=True)
    review_data.drop(['text'], axis=1, inplace=True)

    # Finally, join everything
    training_data = create_evaluation_data(business_data, checkin_data, tip_data, review_data,
                                           tip_features, review_features, training_data)
    training_data.to_csv("processed_data/training_data.csv", index=None)

    # And join the submission data
    submission_data_1 = pd.from_csv('data/SubmissionFormat.csv', index_col=None)
    submission_data_1.ix[:, 'date'] = submission_data_1.ix[:, 'date'].apply(functions.date_to_seconds).astype('int32')
    submission_data_1 = create_evaluation_data(business_data, checkin_data, tip_data, review_data,
                                               tip_features, review_features, submission_data_1)
    submission_data_1.to_csv("processed_data/submission_data.csv", index=None)

    submission_data_2 = pd.from_csv('data/PhaseIISubmissionFormat.csv', index_col=None)
    submission_data_2.ix[:, 'date'] = submission_data_2.ix[:, 'date'].apply(functions.date_to_seconds).astype('int32')
    submission_data_2 = create_evaluation_data(business_data, checkin_data, tip_data, review_data,
                                               tip_features, review_features, submission_data_2)
    submission_data_2.to_csv("processed_data/phase2_data.csv", index=None)

    # TODO wabbit_it!

    return


if __name__ == "__main__":
    main()
