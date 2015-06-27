#!/usr/bin/python3
import functions
import pandas as pd
import numpy as np
import review_lagger
import getopt
import sys


def create_evaluation_data(inspection_data, tip_data, review_data, target_data):
    lagger = review_lagger.Lagger(inspection_data, review_data, tip_data, size=target_data.shape[0])
    df = target_data.apply(lagger, axis=1)
    return df


def main(argv):
    tip_file = 'processed_data/tip_data.csv'
    review_file = 'processed_data/review_data.csv'
    train_file = 'data/train_labels.csv'
    in_files = ['data/train_labels.csv', 'data/SubmissionFormat.csv', 'data/PhaseIISubmissionFormat.csv']
    out_files = ["processed_data/training_data_raw.csv", "processed_data/submission_data_raw.csv",
                 "processed_data/phase2_data_raw.csv"]
    yelp_id_file = "data/restaurant_ids_to_yelp_ids.csv"

    try:
        opts, args = getopt.getopt(argv, "t:r:a:y:i:r:",
                                   ["tip_file=", "review_file=", "train_file=", "yelp_file=", "in_file=",
                                    "out_file="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(1)

    for o, a in opts:
        print("reading %s %s" % (o, a))
        if o in ("-t", "--tip_file"):
            tip_file = a
        elif o in ("-r", "--review_file"):
            review_file = a
        elif o in ("-a", "--train_file"):
            train_file = a
        elif o in ("-y", "--yelp_file"):
            yelp_id_file = a
        elif o in ("-i", "--in_file"):
            in_files = [a]
        elif o in ("-o", "--out_file"):
            out_files = [a]
        else:
            assert False, "unhandled option " + o

    tip_data = pd.DataFrame.from_csv(tip_file, index_col=None)
    review_data = pd.DataFrame.from_csv(review_file, index_col=None)

    # Add restaurant IDs to everything
    id_dict = functions.build_restaurant_id_map(yelp_id_file)
    map_to_boston_ids = lambda yid: id_dict[yid] if yid in id_dict else np.nan

    for df in [tip_data, review_data]:
        df['restaurant_id'] = df['business_id'].map(map_to_boston_ids)
        df.drop(['business_id'], axis=1, inplace=True)
        # drop those businesses that are not in the Boston dataset
        df.dropna(axis=0, subset=['restaurant_id'], inplace=True)
        df.set_index(['restaurant_id', 'date'], inplace=True)
        df.sortlevel(0, inplace=True)

    # Finally, join everything
    training_data = pd.DataFrame.from_csv(train_file, index_col=None)
    training_data.ix[:, 'date'] = training_data.ix[:, 'date'].apply(functions.date_to_seconds).astype('int32')
    training_data.set_index(['restaurant_id', 'date'], inplace=True)
    training_data.sortlevel(0, inplace=True)

    for n, fn in enumerate(in_files):
        in_data = pd.DataFrame.from_csv(fn, index_col=None)
        in_data.ix[:, 'date'] = in_data.ix[:, 'date'].apply(functions.date_to_seconds).astype('int32')
        in_data.set_index(['restaurant_id', 'date'], inplace=True)
        in_data.sortlevel(0, inplace=True)
        out = create_evaluation_data(training_data, tip_data, review_data, in_data)
        out.to_csv(out_files[n], index=None)

    return


if __name__ == "__main__":
    main(sys.argv[1:])
