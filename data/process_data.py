# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):

    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand = True)

    # Use one of the rows to extract a list of names of categories and rename the columns
    row = categories[:1].values
    category_colnames=[]
    for i in range(36):
        category_colnames.append(row[0,i][:-2])
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    #Replace categories column in df with new category columns
    df.drop(['categories'], axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)

    #Remove duplicates
    df.drop_duplicates(inplace=True)

    #Modify the column "related" into "unrelated"
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    df['related']=df['related'].map(lambda x: 1 if x == 0 else 0)
    df.rename(columns={"related":"unrelated"}, inplace=True)

    return df


def save_data(df, database_filename):

    #Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    db_file_name = database_filename.split("/")[-1] # extract file name from \
                                                     # the file path
    table_name = db_file_name.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists = 'replace')

    pass


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
