#!/usr/bin/python3
import argparse
import pandas
import progress.bar
import re
import sklearn.feature_extraction.text

__author__ = 'paste'


class MonitoringTokenizer(object):
    def __init__(self, size, name):
        self._bar = progress.bar.IncrementalBar("Vectorizing %s" % name,
                                                suffix="%(percent)d%% (%(index)d/%(max)d) ETA %(eta_td)s",
                                                max=size)
        self._tick = 0
        self._token_re = re.compile(r"(?u)\b\w\w+\b")

    def __call__(self, doc):
        self._tick += 1
        if self._tick % 100 == 0:
            self._bar.next(100)
        elif self._tick == self._bar.max:
            self._bar.finish()
        return self._token_re.findall(doc)


def main(text_corpus, in_files, out_files):
    df = pandas.DataFrame.from_csv(text_corpus)
    corpus = df['text']
    corpus.fillna("", inplace=True)

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(decode_error="replace",
                                                                 analyzer="word",
                                                                 stop_words="english",
                                                                 ngram_range=(2, 4),
                                                                 lowercase=True,
                                                                 max_df=0.90,
                                                                 min_df=0.01,
                                                                 max_features=1500,
                                                                 sublinear_tf=True,
                                                                 tokenizer=MonitoringTokenizer(corpus.shape[0],
                                                                                               text_corpus))

    vectorizer.fit(corpus)

    vocabulary = vectorizer.get_feature_names()

    del df
    del corpus
    del vectorizer

    for fs in zip(in_files, out_files):
        in_df = pandas.DataFrame.from_csv(fs[0], index_col=None)
        in_corpus = in_df['text']
        in_corpus.fillna("", inplace=True)

        in_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(decode_error="replace",
                                                                        analyzer="word",
                                                                        stop_words="english",
                                                                        ngram_range=(2, 4),
                                                                        lowercase=True,
                                                                        sublinear_tf=True,
                                                                        tokenizer=MonitoringTokenizer(
                                                                            in_corpus.shape[0], fs[0]),
                                                                        vocabulary=vocabulary)

        feature_matrix = in_vectorizer.fit_transform(in_corpus)
        feature_df = pandas.DataFrame(feature_matrix.todense(), columns=vocabulary)
        in_df.drop("text", axis=1)
        out_df = pandas.concat([in_df, feature_df], axis=1)
        print(out_df)
        out_df.set_index('id', inplace=True)
        out_df.to_csv(fs[1])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize the text fields in some data")
    parser.add_argument("-t", "--text-corpus",
                        dest="text_corpus",
                        help="CSV file containing a field labeled \"text\", which will become the corpus to vectorize",
                        metavar="CSVFILE",
                        default="data/training_data_raw.csv")
    parser.add_argument("-i", "--in_file",
                        dest="in_files",
                        nargs="+",
                        help="Input CSV files whose \"text\" field will be updated with the vectorized version",
                        metavar="CSVFILE",
                        default=['data/train_labels_raw.csv',
                                 'data/SubmissionFormat_raw.csv',
                                 'data/PhaseIISubmissionFormat_raw.csv'])
    parser.add_argument("-o", "--out_file",
                        dest="out_files",
                        nargs="+",
                        help="Output file",
                        metavar="CSVFILE",
                        default=["processed_data/training_data_raw_vectorized.csv",
                                 "processed_data/submission_data_raw_vectorized.csv",
                                 "processed_data/phase2_data_raw_vectorized.csv"])

    args = parser.parse_args()

    if len(args.in_files) != len(args.out_files):
        parser.error("number of in_file and out_file arguments must be the same")

    main(text_corpus=args.text_corpus, in_files=args.in_files, out_files=args.out_files)
