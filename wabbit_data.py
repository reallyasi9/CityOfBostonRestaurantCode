#!/usr/bin/python3
import re
import csv
import nltk
import pandas as pd
from progress.bar import Bar


def wabbit_it(df, col_dict, target, outfile):
    with open(outfile, 'w') as of:
        count = 0
        bar = Bar("Wabbitizing %s" % outfile, max=df.shape[0],
                  suffix="%(percent)d%% (%(index)d/%(max)d) ETA %(eta_td)s")
        for row in df:
            count += 1
            if count % 100 == 0:
                bar.next(100)
            print("target %d ID %s", (row[target], row['id']))
            of.write("%0.1f '%d" % (row[target], row['id']))
            for k, a in col_dict.items():
                of.write(" |%s " % k)
                for n, v in row[a].iteritems():
                    of.write("%s:%f " % n, v)
            of.write("\n")
        bar.finish()
    return


def main():
    in_files = ["processed_data/training_data_raw.csv",
                "processed_data/submission_data_raw.csv",
                "processed_data/phase2_data_raw.csv"]
    out_files = ["processed_data/training_data_%d.vw",
                 "processed_data/submission_data_%d.vw",
                 "processed_data/phase2_data_%d.vw"]

    col_dict = {'lagged': ['p*', 'p**', 'p***', 'last*', 'last**', 'last***', 'std*', 'std**', 'std***'],
                "other": ["date", "restaurant_id"],
                "text": ["text"]}

    for fs in zip(in_files, out_files):
        df = pd.DataFrame.from_csv(fs[0], index_col=None)
        for n, tgt in enumerate(['*', '**', '***']):
            wabbit_it(df, col_dict, target=tgt, outfile=fs[1] % n)
    return


if __name__ == "__main__":
    main()
