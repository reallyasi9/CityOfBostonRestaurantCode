'''
'''
from sys import argv
import time

from yelp_class import YelpData


##################################################
#   Main function                                #
##################################################

input_file = argv[0]


def main():
    '''
    Run the program
    '''
    start = time.time()
    test_data = YelpData(df_path=input_file)

    print('starting ...')

    test_data.vectorize_text()
    test_data.export_to_excel('tfidf_output.xlsx')

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
