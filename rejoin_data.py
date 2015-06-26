#!/usr/bin/python3
__author__ = 'paste'

import pandas as pd


def main():
    sub_df = pd.read_table("data/SubmissionFormat.csv", sep=',', index_col='id')
    pre1 = pd.read_table("processed_data/submission_data_0.prediction", sep=' ', index_col='id', header=None,
                         names=['*', 'id'])
    pre2 = pd.read_table("processed_data/submission_data_1.prediction", sep=' ', index_col='id', header=None,
                         names=['**', 'id'])
    pre3 = pd.read_table("processed_data/submission_data_2.prediction", sep=' ', index_col='id', header=None,
                         names=['***', 'id'])

    pre1 = pre1.apply(pd.Series.round)
    pre1 = pre1.clip_lower(0)
    pre2 = pre2.apply(pd.Series.round)
    pre2 = pre2.clip_lower(0)
    pre3 = pre3.apply(pd.Series.round)
    pre3 = pre3.clip_lower(0)

    sub_df['*'] = pre1.astype('int64')
    sub_df['**'] = pre2.astype('int64')
    sub_df['***'] = pre3.astype('int64')
    sub_df.reset_index(inplace=True)
    print(sub_df)

    sub_df.to_csv("phase1_submission.csv", index=None)

    return


if __name__ == "__main__":
    main()
