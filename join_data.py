#!/usr/bin/python3
import argparse
import functions
import pandas as pd
import numpy as np
import review_lagger
import optparse
import sys


def create_evaluation_data(inspection_data, tip_data, review_data, target_data):
    lagger = review_lagger.Lagger(inspection_data, review_data, tip_data, size=target_data.shape[0])
    df = target_data.apply(lagger, axis=1)
    return df


def main(tip_file='processed_data/tip_data.csv',
         review_file='processed_data/review_data.csv',
         train_file='data/train_labels.csv',
         yelp_id_file="data/restaurant_ids_to_yelp_ids.csv",
         business_file="processed_data/business_data.csv",
         checkin_file="processed_data/checkin_data.csv",
         in_files=['data/SubmissionFormat.csv', 'data/train_labels.csv', 'data/PhaseIISubmissionFormat.csv'],
         out_files=["processed_data/submission_data_raw.csv", "processed_data/training_data_raw.csv",
                    "processed_data/phase2_data_raw.csv"]):
    tip_data = pd.DataFrame.from_csv(tip_file, index_col=None)
    review_data = pd.DataFrame.from_csv(review_file, index_col=None)
    business_data = pd.DataFrame.from_csv(business_file, index_col=None)
    checkin_data = pd.DataFrame.from_csv(checkin_file, index_col=None)

    # Add restaurant IDs to everything
    id_dict = functions.build_restaurant_id_map(yelp_id_file)
    map_to_boston_ids = lambda yid: id_dict[yid] if yid in id_dict else np.nan

    for df in [tip_data, review_data, business_data, checkin_data]:
        df['restaurant_id'] = df.loc[:, 'business_id'].map(map_to_boston_ids)
        df.drop(['business_id'], axis=1, inplace=True)
        # drop those businesses that are not in the Boston dataset
        df.dropna(axis=0, subset=['restaurant_id'], inplace=True)

    # Set indices to make the matching faster
    for df in [tip_data, review_data]:
        df.set_index(['restaurant_id', 'date'], inplace=True)
        df.sortlevel(0, inplace=True)

    # Finally, join everything

    # Set up training data
    training_data = pd.DataFrame.from_csv(train_file, index_col=None)
    training_data.loc[:, 'date'] = training_data.loc[:, 'date'].apply(functions.date_to_seconds).astype('int32')
    training_data.set_index(['restaurant_id', 'date'], inplace=True)
    training_data.sortlevel(0, inplace=True)

    # Merge down business and checkin data
    business_data = business_data.groupby("restaurant_id").mean()
    checkin_data = checkin_data.groupby("restaurant_id").mean()

    for fn in zip(in_files, out_files):
        in_data = pd.DataFrame.from_csv(fn[0], index_col=None)
        in_data.loc[:, 'date'] = in_data.loc[:, 'date'].apply(functions.date_to_seconds).astype('int32')
        in_data.set_index(['restaurant_id', 'date'], inplace=True)
        in_data.sortlevel(0, inplace=True)
        out = create_evaluation_data(training_data, tip_data, review_data, in_data)
        out.reset_index(inplace=True)
        out.set_index('restaurant_id', inplace=True)
        out = out.merge(business_data, how="left", left_index=True, right_index=True, sort=False)
        out = out.merge(checkin_data, how="left", left_index=True, right_index=True, sort=False)
        out.reset_index(inplace=True)
        out.loc[:, 'id'] = out.loc[:, 'id'].astype('int32')
        out.set_index('id', inplace=True)
        out.to_csv(fn[1])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join and lag data")
    parser.add_argument("-t", "--tip_file",
                        dest="tip_file",
                        help="File to parse for tip information",
                        metavar="CSVFILE",
                        type=argparse.FileType('r'),
                        default="processed_data/tip_data.csv")
    parser.add_argument("-r", "--review_file",
                        dest="review_file",
                        help="File to parse for review information",
                        metavar="CSVFILE",
                        type=argparse.FileType('r'),
                        default="processed_data/review_data.csv")
    parser.add_argument("-a", "--train_file",
                        dest="train_file",
                        help="Labeled training data set",
                        metavar="CSVFILE",
                        type=argparse.FileType('r'),
                        default="data/train_labels.csv")
    parser.add_argument("-y", "--yid_file",
                        dest="yelp_id_file",
                        help="File containing translation between Yelp! ID and Boston restaurant ID",
                        metavar="JSONFILE",
                        type=argparse.FileType('r'),
                        default="data/restaurant_ids_to_yelp_ids.csv")
    parser.add_argument("-b", "--business_file",
                        dest="business_file",
                        help="Business features data file",
                        metavar="CSVFILE",
                        type=argparse.FileType('r'),
                        default="processed_data/business_data.csv")
    parser.add_argument("-c", "--checkin_file",
                        dest="checkin_file",
                        help="Labeled check-in data file",
                        metavar="CSVFILE",
                        type=argparse.FileType('r'),
                        default="processed_data/checkin_data.csv")
    parser.add_argument("-i", "--in_file",
                        dest="in_files",
                        nargs="+",
                        help="Input file to join with other files",
                        metavar="CSVFILE",
                        type=argparse.FileType('r'),
                        default=['data/train_labels.csv',
                                 'data/SubmissionFormat.csv',
                                 'data/PhaseIISubmissionFormat.csv'])
    parser.add_argument("-o", "--out_file",
                        dest="out_files",
                        nargs="+",
                        help="Output file",
                        metavar="CSVFILE",
                        type=argparse.FileType('w'),
                        default=["processed_data/training_data_raw.csv",
                                 "processed_data/submission_data_raw.csv",
                                 "processed_data/phase2_data_raw.csv"])

    args = parser.parse_args()

    if len(args.in_files) != len(args.out_files):
        parser.error("number of in_file and out_file arguments must be the same")

    main(tip_file=args.tip_file, review_file=args.review_file, train_file=args.train_file,
         yelp_id_file=args.yelp_id_file, business_file=args.business_file,
         checkin_file=args.checkin_file, in_files=args.in_files, out_files=args.out_files)
