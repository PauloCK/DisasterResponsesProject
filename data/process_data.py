import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads datasets and merge them.
    
    Arguments:
        messages_filepath --> Path to the CSV file containing messages
        categories_filepath --> Path to the CSV file containing categories related to the messages 
    Output:
        df --> Merged DataFrame with messages and their respectives categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    """
    Prepares data for the ML process.
    
    Arguments:
        df --> DataFrame with messages and respective categories
    Outputs:
        df --> DataFrame with messages and respective categories prepared for the ML process
    """
    
    # Getting classifications from labeled messages
    categories = df['categories'].str.split(';',expand=True)
    
    #Fixing columns names
    row = categories.iloc[[0]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    # Changing values in the DataFrame
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Saves data to a SQLite database.
    
    Arguments:
        df --> Clean DataFrame with messages and their respective categories
        database_filename --> Destination path to save the SQLite database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    table_name = database_filename.replace("db.db","") + "tbl"
    df.to_sql(table_name, engine, index=False, if_exists='replace')  


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