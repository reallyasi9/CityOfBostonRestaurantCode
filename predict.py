#!/usr/bin/python3
import argparse
import numpy
import pandas
from time import time
import scipy.stats
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import BayesianRidge
from operator import itemgetter
import functions

__author__ = 'paste'

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            numpy.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def main(training_file, in_files, out_files, out_format_files):
    train_df = pandas.DataFrame.from_csv(training_file, index_col="id")
    outcomes = train_df[["*", "**", "***"]]
    train_df = train_df._get_numeric_data()
    train_df.drop(["*", "**", "***"], axis=1, inplace=True)

    # param_map = {"alpha_1": scipy.stats.expon(scale=1.e-6),
    #              "alpha_2": scipy.stats.expon(scale=1.e-6),
    #              "lambda_1": scipy.stats.expon(scale=1.e-6),
    #              "lambda_2": scipy.stats.expon(scale=1.e-6),
    #              "fit_intercept": [True, False],
    #              "normalize": [True, False]}

    # Parameters discovered using RandomizedSearchCV
    # regressor = BayesianRidge(verbose=True,
    #                           alpha_1=1.e-6,
    #                           alpha_2=5.e-7,
    #                           lambda_1=7.e-7,
    #                           lambda_2=8.e-8,
    #                           normalize=False,
    #                           fit_intercept=True)

    # run randomized search
    # n_iter_search = 20
    # random_search = RandomizedSearchCV(regressor, param_distributions=param_map,
    #                                    n_iter=n_iter_search)
    #
    # for oc in ["*", "**", "***"]:
    #     print("Attempting %d searches on %s" % (n_iter_search, oc))
    #     start = time()
    #     random_search.fit(df, outcomes.loc[:, oc])
    #     print("RandomizedSearchCV on %s took %.2f seconds for %d candidates"
    #           " parameter settings." % (oc, (time() - start), n_iter_search))
    #
    #     report(random_search.grid_scores_)

    regressors = {}

    for oc in ["*", "**", "***"]:
        regressors[oc] = BayesianRidge(verbose=True,
                                       alpha_1=1.e-6,
                                       alpha_2=5.e-7,
                                       lambda_1=7.e-7,
                                       lambda_2=8.e-8,
                                       normalize=False,
                                       fit_intercept=True)
        regressors[oc].fit(train_df, outcomes.loc[:, oc])

    del train_df
    del outcomes

    for fs in zip(in_files, out_files, out_format_files):
        predict_df = pandas.DataFrame.from_csv(fs[0], index_col=None)
        out_df = predict_df.loc[:, "id"].astype("int32")
        out_df.set_index("id", inplace=True)
        print(out_df)
        predict_df.set_index("id", inplace=True)
        predict_df = predict_df._get_numeric_data()
        predict_df.drop(["*", "**", "***"], axis="columns", inplace=True)
        for oc in ["*", "**", "***"]:
            predicted_outcome = pandas.DataFrame(regressors[oc].predict(predict_df), columns=[oc],
                                                 index=predict_df.index)
            predicted_outcome = predicted_outcome.apply(pandas.Series.round)
            predicted_outcome = predicted_outcome.clip_lower(0)
            out_df = pandas.concat([out_df, predicted_outcome.astype('int32')], axis="columns")

        out_formatted_df = pandas.DataFrame.from_csv(fs[2], index_col="id")
        out_formatted_df[["*", "**", "***"]] = out_df[["*", "**", "***"]]
        print(out_df)
        print(out_formatted_df)
        out_formatted_df.to_csv(fs[1])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument("-t", "--training-file",
                        dest="training_file",
                        help="File to use for training",
                        metavar="CSVFILE",
                        default="processed_data/training_data_vectorized.csv")
    parser.add_argument("-i", "--in-file",
                        dest="in_files",
                        nargs="+",
                        help="Input file to join with other files",
                        metavar="CSVFILE",
                        default=["processed_data/submission_data_vectorized.csv",
                                 "processed_data/phase2_data_vectorized.csv"])
    parser.add_argument("-o", "--out-file",
                        dest="out_files",
                        nargs="+",
                        help="Output file",
                        metavar="CSVFILE",
                        default=["submission_prediction.csv",
                                 "phase2_prediction.csv"])
    parser.add_argument("-f", "--out-format-file",
                        dest="out_format_files",
                        nargs="+",
                        help="Output format file",
                        metavar="CSVFILE",
                        default=["data/SubmissionFormat.csv",
                                 "data/PhaseIISubmissionFormat.csv"])

    args = parser.parse_args()

    main(args.training_file, args.in_files, args.out_files, args.out_format_files)
