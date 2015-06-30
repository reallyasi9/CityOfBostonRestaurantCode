#!/usr/bin/python3

import argparse
import functions
import numpy as np
import pandas as pd

__author__ = 'pkillewald'


def main(yelp_id_file, business_file, checkin_file, in_files, out_files):
    id_dict = functions.build_restaurant_id_map(yelp_id_file)
    map_to_boston_ids = lambda yid: id_dict[yid] if yid in id_dict else np.nan

    business_data = None
    checkin_data = None

    if business_file is not None:
        business_data = pd.DataFrame.from_csv(business_file, index_col=None)
        business_data['restaurant_id'] = business_data.loc[:, 'business_id'].map(map_to_boston_ids)
        business_data.drop(['business_id'], axis=1, inplace=True)
        # drop those businesses that are not in the Boston dataset
        business_data.dropna(axis=0, subset=['restaurant_id'], inplace=True)
        business_data = business_data.groupby('restaurant_id').mean()
    if checkin_file is not None:
        checkin_data = pd.DataFrame.from_csv(checkin_file, index_col=None)
        checkin_data['restaurant_id'] = checkin_data.loc[:, 'business_id'].map(map_to_boston_ids)
        checkin_data.drop(['business_id'], axis=1, inplace=True)
        # drop those businesses that are not in the Boston dataset
        checkin_data.dropna(axis=0, subset=['restaurant_id'], inplace=True)
        checkin_data = checkin_data.groupby('restaurant_id').mean()

    for fn in zip(in_files, out_files):
        in_data = pd.DataFrame.from_csv(fn[0], index_col='restaurant_id')
        if business_data is not None:
            in_data = in_data.merge(business_data, how="left", left_index=True, right_index=True, sort=False)
            in_data.fillna(0, inplace=True)
        if checkin_data is not None:
            in_data = in_data.merge(checkin_data, how="left", left_index=True, right_index=True, sort=False)
            in_data.fillna(0, inplace=True)
        in_data.reset_index(inplace=True)
        in_data.set_index('id', inplace=True)
        in_data.to_csv(fn[1])

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join and lag data")
    parser.add_argument("-y", "--yid_file",
                        dest="yelp_id_file",
                        help="File containing translation between Yelp! ID and Boston restaurant ID",
                        metavar="JSONFILE",
                        default="data/restaurant_ids_to_yelp_ids.csv")
    parser.add_argument("-b", "--business_file",
                        dest="business_file",
                        help="Business features data file",
                        metavar="CSVFILE")
    parser.add_argument("-c", "--checkin_file",
                        dest="checkin_file",
                        help="Labeled check-in data file",
                        metavar="CSVFILE")
    parser.add_argument("-i", "--in_file",
                        dest="in_files",
                        nargs="+",
                        help="Input file to join with other files",
                        metavar="CSVFILE",
                        default=['data/train_labels.csv',
                                 'data/SubmissionFormat.csv',
                                 'data/PhaseIISubmissionFormat.csv'])
    parser.add_argument("-o", "--out_file",
                        dest="out_files",
                        nargs="+",
                        help="Output file",
                        metavar="CSVFILE",
                        default=["processed_data/training_data_raw.csv",
                                 "processed_data/submission_data_raw.csv",
                                 "processed_data/phase2_data_raw.csv"])

    args = parser.parse_args()

    if len(args.in_files) != len(args.out_files):
        parser.error("number of in_file and out_file arguments must be the same")

    if args.business_file is None and args.checkin_file is None:
        parser.error("at least one of business_file or checkin_file must be specified")

    main(yelp_id_file=args.yelp_id_file, business_file=args.business_file,
         checkin_file=args.checkin_file, in_files=args.in_files, out_files=args.out_files)
