# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import re

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import pickle

import warnings
warnings.filterwarnings('ignore')

def load_data(database_filepath):

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)

    # the column "related" contains value 2, which can lead to error
    # thus changing them to 1
    # df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    # split the data into features(X) and targets(y, column_names)
    X = df.message
    y = df.iloc[:,4:]
    category_names = y.columns

    return X, y, category_names


def tokenize(text):

    # keep only lower case alphabets and numbers in the text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize the words (including lemmatizing and stripping)
    tokens = word_tokenize(text)
    lemmatized = [WordNetLemmatizer().lemmatize(tok).strip() for tok in tokens]

    # remove stop words
    stopwords_ = stopwords.words("english")
    cleantokens = [word for word in lemmatized if word not in stopwords_]

    return cleantokens

def build_model():

    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # MultiOutputClassifier(RandomForestClassifier())

    # hyper-parameter
    parameters = {'clf__estimator__learning_rate': (2.0, 1.0, 0.5)
                  }

    # create model
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):

    y_pred_test = model.predict(X_test)
    test = classification_report(Y_test.values, y_pred_test, target_names = category_names)
    print(test)

    # print accuracy score
    # print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred_test)))

    pass


def save_model(model, model_filepath):

    with open(model_filepath, 'wb') as classifier:
        pickle.dump(model, classifier)

    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        print('Building model...')
        model = build_model()

        print('Training model...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
