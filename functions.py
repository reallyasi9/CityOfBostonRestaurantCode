from numpy import argsort

# a simple way to create a "document" for an inspection is to
# concatenate all the reviews that happened before the inspection date
def flatten_reviews(label_df, reviews):
    """ label_df: inspection dataframe with date, restaurant_id
        reviews: dataframe of reviews

        Returns all of the text of reviews previous to each
        inspection listed in label_df.
    """
    reviews_dictionary = {}

    n = len(label_df)

    for i, (pid, row) in enumerate(label_df.iterrows()):
        # we want to only get reviews for this restaurant that occurred before the inspection
        pre_inspection_mask = (reviews.date < row.date) & (reviews.restaurant_id == row.restaurant_id)

        # pre-inspection reviews
        pre_inspection_reviews = reviews[pre_inspection_mask]

        # join the text
        all_text = ' '.join(pre_inspection_reviews.text)

        # store in dictionary
        reviews_dictionary[pid] = all_text

        if i % 2500 == 0:
            print('{} out of {}'.format(i, n))

    # return series in same order as the original data frame
    return reviews_dictionary


def get_top_features(features, model, level, limit, bottom=False):
    """ Get the top (most likely to see violations) and bottom (least
        likely to see violations) features for a given model.

        :param features: an array of the feature names
        :param model: a fitted linear regression model
        :param level: 0, 1, 2 for *, **, *** violation levels
        :param limit: how many features to return
        :param bottom: if we want the bottom features rather than the top
    """
    # sort order for the coefficients
    sorted_coeffs = argsort(model.coef_[level])

    if bottom:
        # get the features at the end of the sorted list
        return features[sorted_coeffs[-1 * limit:]]
    else:
        # get the features at the beginning of the sorted list
        return features[sorted_coeffs[:limit]]


def build_restaurant_id_map(csvfile):
    """ Build a map between Boston ID and Yelp ID
        :param csvfile: A CSV file containing Boston-to-Yelp ID mappings
        :return a dict containing a mapping between Boston ID and Yelp ID
    """
    id_map = pd.read_csv(csvfile)
    id_dict = {}

    # each Yelp ID may correspond to up to 4 Boston IDs
    for i, row in id_map.iterrows():
        # get the Boston ID
        boston_id = row["restaurant_id"]

        # get the non-null Yelp IDs
        non_null_mask = ~pd.isnull(row.ix[1:])
        yelp_ids = row[1:][non_null_mask].values

        for yelp_id in yelp_ids:
            id_dict[yelp_id] = boston_id

    return id_dict
