import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """Loads message and categories data from csv files.

    Args:
    messages_filepath: string. The path of the csv file to load messages.
    categories_filepath: string. The path of the csv file to load categories.

    Returns:
    df: pandas DataFrame.  Includes all the messages and their categories before cleaning.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df

def clean_data(df):
    """Cleans messages datframe spliting their categories and adapating format.

    Args:
    df: pandas DataFrame.  Includes all the messages and their categories before cleaning.


    Returns:
    df: pandas DataFrame.  Includes all the messages and their categories ready to use. 
    Each category is represented by a column. If the message belongs to a category is 
    representad by 1 and if not is represented by 0.
    """
    categories = df.categories.str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]

    category_colnames = row.apply(lambda x : x.split('-')[0])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x.split('-')[1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')
    for column in categories:
        mask = (categories['related'] != 0) & (categories['related'] != 1)
        categories.loc[mask, column] = int(categories[column].mode())

    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Saves messages DataFrame to the a table in a sqlite database file.
    
    Args:
    df: pandas DataFrame. Includes all the messages and their categories ready to use. 
    database_filename: string. The path where the sqlite database file will be saved.
    
    Returns: None
    """
    engine = create_engine('sqlite:///%s' % database_filename)
    df.to_sql('Messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()