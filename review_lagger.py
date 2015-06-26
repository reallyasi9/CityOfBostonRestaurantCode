import math
import progress.bar

__author__ = 'pkillewald'


class Lagger(object):
    _idf = None
    _rdf = None
    _tdf = None
    _bar = None
    # _tick = 0

    def __init__(self, inspect_df, review_df, tip_df, size=None):
        self._idf = inspect_df[["*", "**", "***"]]
        self._rdf = review_df[["text"]]
        self._tdf = tip_df[["text"]]
        if size is not None:
            self._bar = progress.bar.ChargingBar("Lagging", max=size,
                                                 suffix="%(percent)d%% (%(index)d/%(max)d), ETA %(eta_td)s")

    def __call__(self, row):
        if self._bar is not None:
            # self._tick += 1
            # if self._tick % 100 == 0:
            self._bar.next()
            # elif self._tick == self._bar.max:
            #     self._bar.goto(self._bar.max)

        # Get reviews and tips that contribute to this inspection date
        # (same restaurant, but before now in time)
        rid = row.name[0]
        date = row.name[1]
        
        found_r = self._rdf.ix[rid] if rid in self._rdf.index else None
        found_r = found_r.ix[:date] if found_r is not None else None
        found_t = self._tdf.ix[rid] if rid in self._tdf.index else None
        found_t = found_t.ix[:date] if found_t is not None else None
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
        concat_r = "\n".join(found_r['text'])
        concat_t = "\n".join(found_t['text'])
        row['text'] = "\n".join([concat_r, concat_t])

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
        else:
            wts = prev_rows.index.to_series().add(-date)  # remember: negative!
            wts = wts.map(lambda x: math.exp(x / (4383 * 3600)))  # What?  You don't know how many hours are in a month?
            sum_wts = wts.sum()
            prev_rows.loc[:, 'wt'] = wts / sum_wts

            row['p*'] = prev_rows['*'].mul(prev_rows['wt']).sum()
            row['p**'] = prev_rows['**'].mul(prev_rows['wt']).sum()
            row['p***'] = prev_rows['***'].mul(prev_rows['wt']).sum()
            latest = (prev_rows.index == prev_rows.index.max())
            row['last*'] = prev_rows.loc[latest, '*'].tolist()[0]
            row['last**'] = prev_rows.loc[latest, '**'].tolist()[0]
            row['last***'] = prev_rows.loc[latest, '***'].tolist()[0]
            row['std*'] = prev_rows['*'].std()
            row['std**'] = prev_rows['**'].std()
            row['std***'] = prev_rows['***'].std()

        return row
