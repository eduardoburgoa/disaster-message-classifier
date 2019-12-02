import sys
# import libraries
import pickle
import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from functools import partial

from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """Load data from database.

    Args:
    database_filepath: string. The file path of the sqlite database to load.

    Returns:
    X: pandas Series.  Message series to be used as independent variable for a model.
    Y: pandas DataFrame. Each row represents a message and each column represents the 
    categories used to classify a message.
    category_names: list. Complete list of column names used to classify each message.
    """
    engine = create_engine('sqlite:///%s' % database_filepath)
    df = pd.read_sql_table('Messages', engine)
    category_names = ['related', 'request', 'offer',
        'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
        'security', 'military', 'child_alone', 'water', 'food', 'shelter',
        'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
        'infrastructure_related', 'transport', 'buildings', 'electricity',
        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
        'other_weather', 'direct_report']
    X = df['message']
    Y = df[category_names]
    return X, Y, category_names


def tokenize(text):
    """Split, lemmatize and clean text returning a list of tokens.
    
    Args:
    text: string. The complete text to be tokenized.

    Returns:
    clean_tokens: list. Clean tokens ready to be used for the next step in the nltk pipeline.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Creates the machine learning pipeline.
    
    Args: None.

    Returns:
    model: Pipeline. Pipeline for transform (CountVectorizer and TfidfTransformer) and 
    classify (MultiOutputClassifier).
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=partial(tokenize))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    parameters = parameters = {
        'tfidf__smooth_idf': [True, False],
        'clf__estimator__bootstrap': [True, False]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=8, verbose=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the machine learning model and prints a report with the results.
    
    Args:
    model: Pipeline. The model/pipeline to be evaluated.
    X_test: pandas Series. Message series to be used as independent variable to test the model.
    Y_test: pandas DataFrame. The category classification of the messages to tests the model. 
    Each row represents a message and each column represents the categories used to classify a message.
    category_names:
    
    Returns: None
    """
    Y_pred = model.predict(X_test)

    i = 0
    for column in category_names:
        i += 1
        print(column)
        print(classification_report(Y_test[column], Y_pred[:, 1]))


def save_model(model, model_filepath):
    """Saves the pipeline to a pickle file.
    
    Args:
    model: Pipeline. The pipeline to be saved.
    model_filepath: string. The path where the pickle file will be saved.
    
    Returns: None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
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