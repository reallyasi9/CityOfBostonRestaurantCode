#!/usr/bin/python3
import argparse
import numpy
import pandas
from time import time
import scipy.stats
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import BayesianRidge
from operator import itemgetter

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

def main(training_file):

    df = pandas.DataFrame.from_csv(training_file, index_col="id")
    outcomes = df[["*", "**", "***"]]
    df = df._get_numeric_data()
    df.drop(["*", "**", "***"], axis=1, inplace=True)

    # param_map = {"alpha_1": scipy.stats.expon(scale=1.e-6),
    #              "alpha_2": scipy.stats.expon(scale=1.e-6),
    #              "lambda_1": scipy.stats.expon(scale=1.e-6),
    #              "lambda_2": scipy.stats.expon(scale=1.e-6),
    #              "fit_intercept": [True, False],
    #              "normalize": [True, False]}

    # Parameters discovered using RandomizedSearchCV
    regressor = BayesianRidge(verbose=True,
                              alpha_1=1.e-6,
                              alpha_2=5.e-7,
                              lambda_1=7.e-7,
                              lambda_2=8.e-8,
                              normalize=False,
                              fit_intercept=True)

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
    for oc in ["*", "**", "***"]:
        regressor.fit(df, outcomes.loc[:, oc])
        # TODO Actually predict!


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument("-t", "--training-file",
                        dest="training_file",
                        help="File to use for training",
                        metavar="CSVFILE")

    args = parser.parse_args()

    main(args.training_file)
