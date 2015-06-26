#!/usr/bin/python3
import re
import csv
import pandas as pd
from progress.bar import Bar


def wabbit_it(df, col_dict, target, outfile):
    clean_re = re.compile(r'\W+')
    with open(outfile, 'w') as of:
        count = 0
        bar = Bar("Wabbitizing %s" % outfile, max=df.shape[0],
                  suffix="%(percent)d%% (%(index)d/%(max)d) ETA %(eta_td)s")
        for row in df.iterrows():
            count += 1
            if count % 100 == 0:
                bar.next(100)
            of.write("%0.1f '%d" % (row[1][target], row[1]['id']))
            for k, a in col_dict.items():
                of.write(" |%s " % k)
                for n, v in row[1][a].iteritems():
                    of.write("%s:%f " % (clean_re.sub('_', n), v))
            of.write("\n")
        bar.finish()
    return


if __name__ == "__main__":
    training_data = pd.read_csv('processed_data/training_data.csv')
    submission_data = pd.read_csv('processed_data/submission_data.csv')
    phase2_data = pd.read_csv('processed_data/phase2_data.csv')
    # Can't deal with this now
    # training_data.dropna(axis='columns', inplace=True)
    # submission_data.dropna(axis='columns', inplace=True)
    # phase2_data.dropna(axis='columns', inplace=True)

    business_cols = []
    with open('processed_data/business_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        business_cols = [c for c in reader.fieldnames if c in training_data.columns]
    checkin_cols = []
    with open('processed_data/checkin_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        checkin_cols = [c for c in reader.fieldnames if c in training_data.columns]
    tip_cols = []
    with open('processed_data/tip_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        tip_cols = [c for c in reader.fieldnames if c in training_data.columns]
    review_cols = []
    with open('processed_data/review_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        review_cols = [c for c in reader.fieldnames if c in training_data.columns]

    col_dict = {'business': business_cols,
                'checkin': checkin_cols,
                'tip': tip_cols,
                'review': review_cols}
    for n, tgt in enumerate(['*', '**', '***']):
        wabbit_it(training_data, col_dict, target=tgt,
                  outfile='processed_data/training_data_%d.vw' % n)
        wabbit_it(submission_data, col_dict, target=tgt,
                  outfile='processed_data/submission_data_%d.vw' % n)
        wabbit_it(phase2_data, col_dict, target=tgt,
                  outfile='processed_data/phase2_data_%d.vw' % n)
