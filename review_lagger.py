import math
import re

import nltk
import pandas as pd
import progress.bar

from functions import LemmaTokenizer

__author__ = 'pkillewald'


class Lagger(object):
    _idf = None
    _rdf = None
    _tdf = None
    _bar = None
    _tick = 0
    _stemmer = nltk.stem.snowball.SnowballStemmer("english", ignore_stopwords=True)
    _lemmatizer = LemmaTokenizer()
    _clean_re = re.compile(r'\W+')

    def __init__(self, inspect_df, review_df, tip_df, size=None):
        self._idf = inspect_df[["*", "**", "***"]]
        self._rdf = review_df[["text"]]
        self._tdf = tip_df[["text"]]
        if size is not None:
            self._bar = progress.bar.Bar("Lagging", max=size,
                                         suffix="%(percent)d%% (%(index)d/%(max)d), ETA %(eta_td)s")

    def __call__(self, row):
        if self._bar is not None:
            self._tick += 1
            if self._tick % 100 == 0:
                self._bar.next(100)
            elif self._tick == self._bar.max:
                self._bar.goto(self._bar.max)

        # Get reviews and tips that contribute to this inspection date
        # (same restaurant, but before now in time)
        rid = row.name[0]
        date = row.name[1]

        # Get inspections that happened strictly before this inspection
        prev_rows = self._idf.ix[rid] if rid in self._idf.index else None
        prev_rows = prev_rows.ix[:(date - 1)] if prev_rows is not None else None

        # Generate a weight based on how far in the past the previous inspections happened
        # TODO revisit this model
        # Modeled as an exponential decay of relevance with a time constant of 6 months

        if prev_rows is None or prev_rows.shape[0] == 0:
            # FIXME Do something smarter than zeros?
            row['p*'] = 0
            row['p**'] = 0
            row['p***'] = 0
            row['last*'] = 0
            row['last**'] = 0
            row['last***'] = 0
            row['std*'] = 0
            row['std**'] = 0
            row['std***'] = 0
            return row

        else:

            latest = (prev_rows.index == prev_rows.index.max())
            prev_date = prev_rows[latest].index[0]

            found_r = self._rdf.ix[rid] if rid in self._rdf.index else None
            found_r = found_r.ix[prev_date:date] if found_r is not None else None
            found_t = self._tdf.ix[rid] if rid in self._tdf.index else None
            found_t = found_t.ix[prev_date:date] if found_t is not None else None
            if found_r is None or found_t is None:
                # FIXME Do something smarter than zeros?
                row['p*'] = 0
                row['p**'] = 0
                row['p***'] = 0
                row['last*'] = 0
                row['last**'] = 0
                row['last***'] = 0
                row['std*'] = 0
                row['std**'] = 0
                row['std***'] = 0
                return row

            # Simply concatenate the reviews and tips together
            concat_r = " ".join(found_r['text'])
            concat_t = " ".join(found_t['text'])
            found_cat = " ".join([concat_r, concat_t])
            row['text'] = " ".join(self._lemmatizer(found_cat))

            wts = prev_rows.index.to_series().add(-date)  # remember: negative!
            wts = wts.map(
                lambda x: math.exp(x / (4383 * 3600)))  # What?  You don't know how many hours are in 6 months?
            sum_wts = wts.sum()
            prev_rows.loc[:, 'wt'] = wts / sum_wts

            row['p*'] = prev_rows['*'].mul(prev_rows['wt']).sum()
            row['p**'] = prev_rows['**'].mul(prev_rows['wt']).sum()
            row['p***'] = prev_rows['***'].mul(prev_rows['wt']).sum()
            row['last*'] = prev_rows.loc[latest, '*'].tolist()[0]
            row['last**'] = prev_rows.loc[latest, '**'].tolist()[0]
            row['last***'] = prev_rows.loc[latest, '***'].tolist()[0]
            if prev_rows.shape[0] == 1:
                row['std*'] = row['last*']
                row['std**'] = row['last**']
                row['std***'] = row['last***']
            else:
                row['std*'] = prev_rows['*'].std()
                row['std**'] = prev_rows['**'].std()
                row['std***'] = prev_rows['***'].std()

        return row
