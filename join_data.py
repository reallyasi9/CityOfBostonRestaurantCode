#!/usr/bin/python3
import functions
import pandas as pd
import numpy as np
import review_lagger

def create_evaluation_data(inspection_data, tip_data, review_data, target_data):
    lagger = review_lagger.Lagger(inspection_data, review_data, tip_data, size=target_data.shape[0])
    df = target_data.apply(lagger, axis=1)
    return df


def main():
    tip_data = pd.DataFrame.from_csv('processed_data/tip_data.csv', index_col=None)
    review_data = pd.DataFrame.from_csv('processed_data/review_data.csv', index_col=None)

    # Add restaurant IDs to everything
    id_dict = functions.build_restaurant_id_map('data/restaurant_ids_to_yelp_ids.csv')
    map_to_boston_ids = lambda yid: id_dict[yid] if yid in id_dict else np.nan

    for df in [tip_data, review_data]:
        df['restaurant_id'] = df['business_id'].map(map_to_boston_ids)
        df.drop(['business_id'], axis=1, inplace=True)
        # drop those businesses that are not in the Boston dataset
        df.dropna(axis=0, subset=['restaurant_id'], inplace=True)
        df.set_index(['restaurant_id', 'date'], inplace=True)
        df.sortlevel(0, inplace=True)

    # Finally, join everything
    training_data = pd.DataFrame.from_csv('data/train_labels.csv', index_col=None)
    training_data.ix[:, 'date'] = training_data.ix[:, 'date'].apply(functions.date_to_seconds).astype('int32')
    training_data.set_index(['restaurant_id', 'date'], inplace=True)
    training_data.sortlevel(0, inplace=True)
    train_out = create_evaluation_data(training_data, tip_data, review_data, training_data)
    train_out.to_csv("processed_data/training_data_raw.csv", index=None)

    # And join the submission data
    submission_data_1 = pd.from_csv('data/SubmissionFormat.csv', index_col=None)
    submission_data_1.ix[:, 'date'] = submission_data_1.ix[:, 'date'].apply(functions.date_to_seconds).astype('int32')
    submission_data_1.set_index(['restaurant_id', 'date'], inplace=True)
    submission_data_1.sortlevel(0, inplace=True)
    submission_data_1 = create_evaluation_data(training_data, tip_data, review_data, submission_data_1)
    submission_data_1.to_csv("processed_data/submission_data_raw.csv", index=None)

    submission_data_2 = pd.from_csv('data/PhaseIISubmissionFormat.csv', index_col=None)
    submission_data_2.ix[:, 'date'] = submission_data_2.ix[:, 'date'].apply(functions.date_to_seconds).astype('int32')
    submission_data_2.set_index(['restaurant_id', 'date'], inplace=True)
    submission_data_2.sortlevel(0, inplace=True)
    submission_data_2 = create_evaluation_data(training_data, tip_data, review_data, submission_data_2)
    submission_data_2.to_csv("processed_data/phase2_data_raw.csv", index=None)

    return


if __name__ == "__main__":
    main()
