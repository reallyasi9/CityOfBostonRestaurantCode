#!/usr/bin/python3
import argparse
import pandas
import pickle
import sklearn

__author__ = 'paste'


def main(in_files, out_files, out_format_files, classifier_file, regressor_files, classifier_columns,
         regressor_columns):
    targets = ["*", "**", "***"]

    with open(classifier_file, "rb") as cf:
        classifier = pickle.load(cf)

    for files in zip(in_files, out_files, out_format_files):

        df = pandas.DataFrame.from_csv(files[0], index_col="id")
        if classifier_columns is None:
            classify_df = df._get_numeric_data().astype("float64")
            classify_df.drop(targets, inplace=True, axis="columns")
        else:
            classify_df = df.loc[:, classifier_columns].astype("float64")
        classify_df = classify_df.apply(lambda x: sklearn.preprocessing.StandardScaler().fit_transform(x))

        violations = classifier.predict(classify_df)

        # for n, target in enumerate(targets):
        target = targets
        n = 0
        with open(regressor_files[n], "rb") as rf:
            regressor = pickle.load(rf)

        if regressor_columns is None:
            regressor_df = df._get_numeric_data().astype("float64")
            regressor_df.drop(targets, inplace=True, axis="columns")
        else:
            regressor_df = df.loc[:, regressor_columns].astype("float64")
        regressor_df = regressor_df.apply(lambda x: sklearn.preprocessing.StandardScaler().fit_transform(x))

        predicted_outcome = regressor.predict(regressor_df)

        df.loc[violations, target] = predicted_outcome

        df.loc[(violations == 0), target] = 0
        df[targets] = df[targets].apply(pandas.Series.round)
        df[targets] = df[targets].clip_lower(0)

        out_formatted_df = pandas.DataFrame.from_csv(files[2], index_col="id")
        out_formatted_df[targets] = df[targets].astype("int64")
        out_formatted_df.to_csv(files[1])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
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
    parser.add_argument("-c", "--classifier",
                        dest="classifier",
                        help="Classifier Pickle file",
                        metavar="PKLFILE",
                        default="trained/RandomForestClassifier.pkl")
    parser.add_argument("-r", "--regressors",
                        dest="regressors",
                        help="Regressor Pickle files",
                        nargs="+",
                        metavar="PKLFILE",
                        default=["trained/MultiTaskElasticNetRegressor.pkl"])
    parser.add_argument("-cc", "--classifier-column",
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
    parser.add_argument("-rc", "--regressor-column",
                        dest="regressor_columns",
                        help="Columns to use in regression",
                        nargs="+",
                        default=None)

    args = parser.parse_args()

    main(args.in_files, args.out_files, args.out_format_files, args.classifier, args.regressors,
         args.classifier_columns, args.regressor_columns)
