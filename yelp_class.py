"""
All Yelp, all the time
"""
from os import path, getcwd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np


##################################################
#   Global variables                             #
##################################################


# current working directory + 'data'
WORKING_DIR = path.join(getcwd(), 'data')

# path to Yelp ID / Business ID CSV
RESTAURANT_TO_BIZ_CSV = path.join(WORKING_DIR, 'restaurant_ids_to_yelp_ids.csv')

# path training labels
TRAINING_LABELS_CSV = path.join(WORKING_DIR, 'train_labels.csv')

# json files
JSON_DATA = [('business', 'yelp_academic_dataset_business.json'),
             # ('checkin', 'yelp_academic_dataset_checkin.json'),
             ('review', 'yelp_academic_dataset_review_old.json'),
             # ('user', 'yelp_academic_dataset_user.json'),
             ('tip', 'yelp_academic_dataset_tip.json')]

# json file names and paths dictionary
JSON_NAMES_PATHS_DICT = {i: path.join(WORKING_DIR, j) for i, j in JSON_DATA}


##################################################
#   Helper functions                             #
##################################################


def make_rest_to_biz_dict(rest_to_biz_csv):
    """
    Create a Restaurant ID-to-Business ID dict

    :param rest_to_biz_csv: CSV mapping biz IDs to restaurant IDs
    """
    restaurant_to_biz_df = pd.read_csv(rest_to_biz_csv)
    restaurant_to_biz_dict = {}

    # iterate through rows in dataframe
    for i, row in restaurant_to_biz_df.iterrows():
        target_restaurant_id = row.restaurant_id

        # create (T/F) mask to indicate non-nulls
        non_null_mask = pd.notnull(row.ix[1:])
        target_biz_ids = row[1:][non_null_mask].values

        # iterate through Yelp IDs
        for biz_id in target_biz_ids:
            restaurant_to_biz_dict[biz_id] = target_restaurant_id

    return restaurant_to_biz_dict


def get_target_json(json_file_path):
    """
    Read JSON into dataframe

    :param json_file_path: path to JSON file
    """

    # json_df = None

    try:

        with open(json_file_path, 'r') as opened_json:

            # add brackets to make dict array
            target_json = '[' + ','.join(opened_json.readlines()) + ']'

            # read json into dataframe
            json_df = pd.read_json(target_json, dtype=str)

    except:
        raise Exception('JSON file ' + json_file_path + ' not valid.')

    return json_df


def create_target_dataframe(json_df, biz_id_dict):
    """
    Convert JSON into dataframe

    :param json_df:
    :param biz_id_dict:
    """

    def get_restaurant_id(biz_id):
        """
        Swap business ID for restaurant ID;
        if ID does not exist, make restaurant ID nan.

        :param biz_id:
        """
        if biz_id in biz_id_dict:
            restaurant_to_biz_id_map = biz_id_dict[biz_id]

        else:
            restaurant_to_biz_id_map = np.nan

        return restaurant_to_biz_id_map

    if 'business_id' in json_df.columns:
        # swap business ID for restaurant ID
        json_df.business_id = json_df.business_id.map(get_restaurant_id)

        # rename business_id restaurant_id
        json_df = json_df.rename(columns={'business_id': 'restaurant_id'})

        # get non-null values
        json_df = json_df[pd.notnull(json_df.restaurant_id)]

    return json_df


##################################################
#   Lemmatizer Class                             #
##################################################


class LemmaTokenizer(object):
    """
    This is a class to tokenize using WordNet Lemmatizer
    for use with TFIDF Vectorizer.
    """

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


##################################################
#   YelpData Class                               #
##################################################


class YelpData(object):
    """
    Extracted data for Yelp modeling
    """

    # dictionary of dataframes from JSON files
    data_dict = {}

    # dictionary of restaurant to biz IDs
    biz_id_dict = make_rest_to_biz_dict(RESTAURANT_TO_BIZ_CSV)

    # training labels
    train_labels = pd.read_csv(TRAINING_LABELS_CSV)

    # add json files to data_dict
    for file_name, file_path in JSON_NAMES_PATHS_DICT.items():
        dataframe = create_target_dataframe(get_target_json(file_path),
                                            biz_id_dict)
        data_dict[file_name] = dataframe

    def __init__(self, df_path=None):
        """
        :param df_path: path to CSV or Excel file with current
                           dataframe output
        """
        self.train_and_biz_merged = False
        self.train_and_check_merged = False
        self.text_vectorizer = None
        self.vectorize = None

        self.current_df = df_path

        if df_path:

            # read CSV into dataframe
            if df_path.endswith('.csv'):

                self.current_df = pd.read_csv(df_path)
                self.train_and_biz_merged = True

            # read Excel into dataframe
            elif df_path.endswith('.xls') or df_path.endswith('.xlsx'):

                self.current_df = pd.read_excel(df_path)
                self.train_and_biz_merged = True

            # invalid path or file extension
            else:
                raise Exception('The current_df_path argument must be\
                                 a valid path to a CSV or an Excel file.')

    def __str__(self):
        """
        print current dataframe
        """
        return self.current_df

    def merge_train(self, business=True):
        """
        Merge training data and business data.
        """
        if business and self.train_and_biz_merged is False:
            target_df = YelpData.data_dict['business']
            self.train_and_biz_merged = True
            self.current_df = YelpData.train_labels.merge(target_df,
                                                          on='restaurant_id')

        elif business is False:
            target_df = YelpData.data_dict['checkin']
            target_df.drop(['type'], inplace=True, axis=1)
            self.train_and_check_merged = True
            self.current_df = self.current_df.merge(target_df,
                                                    on='restaurant_id')

        return self.current_df

    def add_feature(self, func):
        """
        :param func: function to apply
        """
        if self.train_and_biz_merged is False:
            self.merge_train()

        result = self.current_df.apply(func, axis=1)
        self.current_df = self.current_df.merge(result,
                                                left_index=True,
                                                right_index=True)

        return self.current_df

    def flatten_reviews_and_tips(self):
        """
        Get tip and review text
        """
        # get review & tip data, and group by restaurant ID
        reviews = YelpData.data_dict['review'].groupby('restaurant_id')
        tips = YelpData.data_dict['tip'].groupby('restaurant_id')

        # count begins at zero
        count = [0]

        def flatten_each_row(row):
            """
            TODO: Refactor this code??
            """

            def get_text(r, df):
                """
                """
                text, user = '', ''

                if r.restaurant_id in df.groups:
                    t = df.get_group(r.restaurant_id)
                    t = t[t.date < r.date]

                    text = t.text.str.cat(sep=' ')
                    user = t.user_id.str.cat(sep='||')

                return text, user

            ftext, fuser = zip(get_text(row, reviews), get_text(row, tips))
            ftext, fuser = ''.join(ftext), ''.join(fuser)

            count[0] += 1
            if count[0] % 2500 == 0:
                print(count[0])

            return pd.Series({'review_tip_text': ftext,
                              'review_tip_user': fuser})

        self.add_feature(flatten_each_row)

        return self.current_df

    def vectorize_text(self, text=None, **kwargs):
        """
        This function vectorizes the data using TFIDF
        This code could probably be refactored.
        """

        if not self.text_vectorizer:
            text = self.current_df.fillna('').review_tip_text

            if kwargs:
                self.text_vectorizer = TfidfVectorizer(**kwargs)

            else:
                print('...initialize TFIDF')
                self.text_vectorizer = TfidfVectorizer(stop_words='english',
                                                       decode_error='replace',
                                                       analyzer='word',
                                                       ngram_range=(1, 2),
                                                       max_features=1000,
                                                       tokenizer=LemmaTokenizer())

            print('...start to transform')
            data = self.text_vectorizer.fit_transform(text)
            print('...done with transform')
            print(self.text_vectorizer.get_feature_names())
            data = pd.DataFrame(data.todense())
            data.columns = self.text_vectorizer.get_feature_names()

        else:
            data = self.text_vectorizer.transform(text)
            data = pd.DataFrame(data.todense())

        self.current_df = data

        return self.current_df

    def get_check_in_and_attributes(self):
        """
        Get check-in information and attributes
        and convert to individual columns.

        This only goes one level down.

        TODO: UPDATE FOR NESTED DICTS!
        """
        if self.train_and_biz_merged is False:
            self.merge_train()

        # if self.train_and_check_merged is False:
        #    self.merge_train(business=False)

        def get_atts_and_check(row):
            """
            Apply function to get attributes and check-in
            dicts, merge them and return a series.
            """
            if row['Ambience'] is not np.nan:
                attributes = ast.literal_eval(row['Ambience'])  # .attributes)
            else:
                attributes = {}
                # if row['Parking'] is not np.nan:
                # checkin = ast.literal_eval(row['Parking']) #.checkin_info)
                # else:
                # checkin = {}

            # attributes.update(checkin)

            return pd.Series(attributes)

        self.add_feature(get_atts_and_check)

        return self.current_df

    def get_categories(self):
        """
        TODO: UPDATE THIS FUNCTION!!
        """
        if self.train_and_biz_merged is False:
            self.merge_train()

        self.current_df['neighborhoods'] = self.current_df.apply(lambda x: ast.literal_eval(x['neighborhoods']), axis=1)
        all_categories = set([c for l in self.current_df['neighborhoods'].values for c in l])
        for category in all_categories:
            self.current_df[category] = self.current_df.apply(lambda x: (1 if category in x['neighborhoods'] else 0),
                                                              axis=1)

        return self.current_df

    def do_it_all(self):
        """
        Start from scratch and do it all

        TODO: UPDATE THIS FUNCTION!!
        """
        return self.current_df

    def export_to_excel(self, excel_file_path):
        """
        Export current dataframe to Excel

        :param excel_file_path: path to excel output file
        """
        if self.current_df is not None:
            # target_columns = self.current_df.select_dtypes(include=[object]).columns
            # self.current_df[target_columns] = self.current_df[target_columns]
            self.current_df.to_excel(excel_file_path)

        return self.current_df

    def get_columns(self):
        """
        Get columns from current_df
        """
        if self.current_df is not None:
            return self.current_df.columns

        return self.current_df


##################################################
#   Main function                                #
##################################################


def main():
    """
    Run the program
    """
    test_data = YelpData()

    print('starting to generate feature ...')

    # add test feature, multiple '*' by 10
    test_data.add_feature(lambda x: x['*'] * 10, 'test_feature')

    print('finished generating feature...')

    # print data
    print(test_data.current_df.head())


if __name__ == '__main__':
    main()
