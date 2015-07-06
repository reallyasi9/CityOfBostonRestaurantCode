#!/usr/bin/python3
import argparse
import numpy
import pandas
from time import time
import pickle
import scipy.stats
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter
from sklearn.linear_model import MultiTaskElasticNet
import sklearn.tree

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


def main(training_file, classifier_columns, regressor_columns, classifier_outfile, regressor_outfiles):
    targets = ["*", "**", "***"]

    train_df = pandas.DataFrame.from_csv(training_file, index_col="id")
    outcomes = train_df.loc[:, targets]

    if classifier_columns is None:
        classify_df = train_df._get_numeric_data().astype("float64")
        classify_df.drop(targets, inplace=True, axis="columns")
    else:
        classify_df = train_df.loc[:, classifier_columns].astype("float64")
    classify_df = classify_df.apply(lambda x: sklearn.preprocessing.StandardScaler().fit_transform(x))

    classifier = RandomForestClassifier(n_jobs=-1,
                                        verbose=1)

    bdt_param_map = {"n_estimators": scipy.stats.randint(low=1565, high=1566),
                     "min_samples_leaf": scipy.stats.randint(low=2, high=3)}

    # Randomize classification
    n_iter_search = 1
    random_search = RandomizedSearchCV(classifier,
                                       param_distributions=bdt_param_map,
                                       n_iter=n_iter_search,
                                       refit=True)

    print("Attempting %d searches" % n_iter_search)
    clamped_outcomes = outcomes.any(axis="columns")
    start = time()
    random_search.fit(classify_df, clamped_outcomes)
    print("Search took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))

    report(random_search.grid_scores_)
    with open(classifier_outfile, "wb") as of:
        pickle.dump(random_search.best_estimator_, of)

    violators = random_search.best_estimator_.predict(classify_df)

    if regressor_columns is None:
        regress_df = train_df._get_numeric_data().astype("float64")
        regress_df.drop(targets, inplace=True, axis="columns")
    else:
        regress_df = train_df.loc[:, regressor_columns].astype("float64")
    regress_df = regress_df[violators]
    regress_df = regress_df.apply(lambda x: sklearn.preprocessing.StandardScaler().fit_transform(x))

    regressor = MultiTaskElasticNet()

    float_outcomes = outcomes.astype("float64")

    n_iter_search = 50
    bdt_param_map = {"alpha": scipy.stats.uniform(loc=.5, scale=10),
                     "l1_ratio": scipy.stats.uniform()}

    # Randomize classification

    # for n, oc in enumerate(targets):
    # Randomize classification
    random_search = RandomizedSearchCV(regressor,
                                       param_distributions=bdt_param_map,
                                       n_iter=n_iter_search,
                                       refit=True)

    oc = targets
    print("Attempting %d searches on %s" % (n_iter_search, oc))
    start = time()
    random_search.fit(regress_df, float_outcomes.loc[violators, oc])
    print("RandomizedSearchCV on %s took %.2f seconds for %d candidates"
          " parameter settings." % (oc, (time() - start), n_iter_search))

    report(random_search.grid_scores_)

    with open(regressor_outfiles[0], "wb") as of:
        pickle.dump(random_search.best_estimator_, of)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument("-t", "--training-file",
                        dest="training_file",
                        help="File to use for training",
                        metavar="CSVFILE",
                        default="processed_data/training_data_vectorized.csv")
    parser.add_argument("-c", "--classifier-column",
                        dest="classifier_columns",
                        help="Columns to use in classification",
                        nargs="+",
                        default=['date', 'last*', 'last**', 'last***', 'p*', 'p**', 'p***', 'std*', 'std**', 'std***',
                                 'attributes.Accepts Credit Cards', 'attributes.Ambience.casual',
                                 'attributes.Ambience.classy', 'attributes.Ambience.divey',
                                 'attributes.Ambience.hipster', 'attributes.Ambience.intimate',
                                 'attributes.Ambience.romantic', 'attributes.Ambience.touristy',
                                 'attributes.Ambience.trendy', 'attributes.Ambience.upscale', 'attributes.BYOB',
                                 'attributes.By Appointment Only', 'attributes.Caters', 'attributes.Coat Check',
                                 'attributes.Corkage', 'attributes.Delivery',
                                 'attributes.Dietary Restrictions.dairy-free',
                                 'attributes.Dietary Restrictions.gluten-free', 'attributes.Dietary Restrictions.halal',
                                 'attributes.Dietary Restrictions.kosher', 'attributes.Dietary Restrictions.soy-free',
                                 'attributes.Dietary Restrictions.vegan',
                                 'attributes.Dietary Restrictions.vegetarian', 'attributes.Dogs Allowed',
                                 'attributes.Drive-Thru', 'attributes.Good For Dancing', 'attributes.Good For Groups',
                                 'attributes.Good For Kids', 'attributes.Good For.breakfast',
                                 'attributes.Good For.brunch', 'attributes.Good For.dessert',
                                 'attributes.Good For.dinner', 'attributes.Good For.latenight',
                                 'attributes.Good For.lunch', 'attributes.Good for Kids', 'attributes.Happy Hour',
                                 'attributes.Has TV', 'attributes.Music.background_music', 'attributes.Music.dj',
                                 'attributes.Music.jukebox', 'attributes.Music.karaoke', 'attributes.Music.live',
                                 'attributes.Music.video', 'attributes.Open 24 Hours', 'attributes.Order at Counter',
                                 'attributes.Outdoor Seating', 'attributes.Parking.garage', 'attributes.Parking.lot',
                                 'attributes.Parking.street', 'attributes.Parking.valet',
                                 'attributes.Parking.validated', 'attributes.Payment Types.amex',
                                 'attributes.Payment Types.cash_only', 'attributes.Payment Types.discover',
                                 'attributes.Payment Types.mastercard', 'attributes.Payment Types.visa',
                                 'attributes.Price Range', 'attributes.Smoking', 'attributes.Take-out',
                                 'attributes.Takes Reservations', 'attributes.Waiter Service',
                                 'attributes.Wheelchair Accessible', 'categories.Active Life',
                                 'categories.Adult Entertainment', 'categories.Afghan',
                                 'categories.African', 'categories.American (New)', 'categories.American (Traditional)',
                                 'categories.Art Galleries', 'categories.Arts & Entertainment',
                                 'categories.Asian Fusion', 'categories.Austrian', 'categories.Bagels',
                                 'categories.Bakeries', 'categories.Bangladeshi', 'categories.Barbeque',
                                 'categories.Bars', 'categories.Basque', 'categories.Beer, Wine & Spirits',
                                 'categories.Belgian', 'categories.Books, Mags, Music & Video', 'categories.Bookstores',
                                 'categories.Bowling', 'categories.Brazilian', 'categories.Breakfast & Brunch',
                                 'categories.Breweries', 'categories.British', 'categories.Bubble Tea',
                                 'categories.Buffets', 'categories.Burgers', 'categories.Burmese', 'categories.Cafes',
                                 'categories.Cajun/Creole', 'categories.Cambodian', 'categories.Cantonese',
                                 'categories.Caribbean', 'categories.Caterers', 'categories.Cheese Shops',
                                 'categories.Cheesesteaks', 'categories.Chicken Wings', 'categories.Chinese',
                                 'categories.Chocolatiers & Shops', 'categories.Cocktail Bars',
                                 'categories.Coffee & Tea', 'categories.Colleges & Universities',
                                 'categories.Colombian', 'categories.Comedy Clubs', 'categories.Comfort Food',
                                 'categories.Convenience Stores', 'categories.Creperies', 'categories.Cuban',
                                 'categories.Cupcakes', 'categories.Dance Clubs', 'categories.Delis',
                                 'categories.Desserts', 'categories.Dim Sum', 'categories.Diners',
                                 'categories.Dive Bars', 'categories.Do-It-Yourself Food', 'categories.Dominican',
                                 'categories.Donuts', 'categories.Education', 'categories.Educational Services',
                                 'categories.Ethiopian', 'categories.Ethnic Food',
                                 'categories.Event Planning & Services', 'categories.Falafel', 'categories.Fashion',
                                 'categories.Fast Food', 'categories.Fish & Chips', 'categories.Fondue',
                                 'categories.Food', 'categories.Food Delivery Services', 'categories.Food Stands',
                                 'categories.Food Trucks', 'categories.French', 'categories.Fruits & Veggies',
                                 'categories.Gastropubs', 'categories.Gay Bars', 'categories.Gelato',
                                 'categories.German', 'categories.Gluten-Free', 'categories.Greek',
                                 'categories.Grocery', 'categories.Halal', 'categories.Health Markets',
                                 'categories.Himalayan/Nepalese', 'categories.Hookah Bars', 'categories.Hot Dogs',
                                 'categories.Hot Pot', 'categories.Hotels', 'categories.Hotels & Travel',
                                 'categories.Hungarian', 'categories.Ice Cream & Frozen Yogurt', 'categories.Indian',
                                 'categories.Internet Cafes', 'categories.Irish', 'categories.Irish Pub',
                                 'categories.Italian', 'categories.Japanese', 'categories.Jazz & Blues',
                                 'categories.Juice Bars & Smoothies', 'categories.Karaoke', 'categories.Korean',
                                 'categories.Kosher', 'categories.Latin American', 'categories.Lebanese',
                                 'categories.Live/Raw Food', 'categories.Lounges', 'categories.Malaysian',
                                 'categories.Meat Shops', 'categories.Mediterranean', "categories.Men's Clothing",
                                 'categories.Mexican', 'categories.Middle Eastern', 'categories.Modern European',
                                 'categories.Mongolian', 'categories.Moroccan', 'categories.Museums',
                                 'categories.Music Venues', 'categories.Nightlife', 'categories.Pakistani',
                                 'categories.Party & Event Planning', 'categories.Performing Arts',
                                 'categories.Persian/Iranian', 'categories.Peruvian', 'categories.Pizza',
                                 'categories.Polish', 'categories.Pool Halls', 'categories.Portuguese',
                                 'categories.Pubs', 'categories.Puerto Rican', 'categories.Ramen',
                                 'categories.Restaurants', 'categories.Salad', 'categories.Sandwiches',
                                 'categories.Scottish', 'categories.Seafood', 'categories.Seafood Markets',
                                 'categories.Senegalese', 'categories.Shopping', 'categories.Soul Food',
                                 'categories.Soup', 'categories.Southern', 'categories.Spanish',
                                 'categories.Specialty Food', 'categories.Sports Bars', 'categories.Steakhouses',
                                 'categories.Sushi Bars', 'categories.Szechuan', 'categories.Taiwanese',
                                 'categories.Tapas Bars', 'categories.Tapas/Small Plates', 'categories.Tea Rooms',
                                 'categories.Tex-Mex', 'categories.Thai', 'categories.Tobacco Shops',
                                 'categories.Turkish', 'categories.Vegan', 'categories.Vegetarian',
                                 'categories.Venezuelan', 'categories.Venues & Event Spaces', 'categories.Vietnamese',
                                 'categories.Wine Bars', 'categories.Wineries', "categories.Women's Clothing",
                                 'latitude', 'longitude', 'neighborhoods.Allston/Brighton', 'neighborhoods.Back Bay',
                                 'neighborhoods.Beacon Hill', 'neighborhoods.Central Square',
                                 'neighborhoods.Charlestown', 'neighborhoods.Chinatown', 'neighborhoods.Dorchester',
                                 'neighborhoods.Downtown', 'neighborhoods.Dudley Square', 'neighborhoods.East Boston',
                                 'neighborhoods.Egleston Square', 'neighborhoods.Fenway', 'neighborhoods.Fields Corner',
                                 'neighborhoods.Financial District', 'neighborhoods.Hyde Park',
                                 'neighborhoods.Jamaica Plain', 'neighborhoods.Leather District',
                                 'neighborhoods.Mattapan', 'neighborhoods.Mission Hill', 'neighborhoods.North End',
                                 'neighborhoods.Roslindale', 'neighborhoods.Roslindale Village',
                                 'neighborhoods.South Boston', 'neighborhoods.South End', 'neighborhoods.Uphams Corner',
                                 'neighborhoods.Waterfront', 'neighborhoods.West Roxbury',
                                 'neighborhoods.West Roxbury Center', 'review_count', 'stars', 'zip'])
    parser.add_argument("-r", "--regressor-column",
                        dest="regressor_columns",
                        help="Columns to use in regression",
                        nargs="+",
                        default=None)
    parser.add_argument("-co", "--classifier-outfile",
                        dest="classifier_outfile",
                        help="Outout file for classifier",
                        default="trained/RandomForestClassifier.pkl")
    parser.add_argument("-ro", "--regressor-outfile",
                        dest="regressor_outfiles",
                        help="Outout files for regressors",
                        nargs="+",
                        default=["trained/MultiTaskElasticNetRegressor.pkl"])

    args = parser.parse_args()

    main(args.training_file, args.classifier_columns, args.regressor_columns, args.classifier_outfile,
         args.regressor_outfiles)
