import os
import pickle
import re
from sqlalchemy import create_engine
import sys

import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


nltk.download(['averaged_perceptron_tagger','punkt', 'stopwords', 'wordnet'])

def load_data(db_name, tbl_name):
    """
    Loads data from a SQLite table.
    
    Arguments:
        db_name --> Name of the SQLite database
        tbl_name --> Name of the table containing the data
    Output:
        X --> Features DataFrame
        Y --> Labels DataFrame
        categories --> Possible categories in which messages in `X` can fit 
    """
    
    engine = create_engine(f'sqlite:///{db_name}')
    
    df = pd.read_sql_table(tbl_name, engine)
    X = df['message']
    y = df.iloc[:,4:]
    
    # There are only a few rows that have the value of 2, here we are
    # supposing that those values should be 1 and assigning this value to them
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    
    # Extracting the possible categories of the messages
    categories = y.columns
    
    return X, y, categories

def tokenize(text,url_placeholder='url_placeholder'):
    """
    Create a list of tokens from `text` using `url_placeholder` to
    replace any URL present in `text`.
    
    Arguments:
        text --> Text to be tokenized.
        url_placeholder --> A placeholder for any URL present in `text`.
    Output:
        clean_tokens --> List of tokens created based on `text`.
    """
    
    # Regex string to match an URL
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detecting URL's in `text`
    detected_urls = re.findall(url_regex, text)
    
    # Replacing URL's present in the `text` with `url_placeholder`
    for url in detected_urls:
        text = text.replace(url, url_placeholder)
        
    # Removing punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        
    # Tokenizing `text`
    tokens = nltk.word_tokenize(text)
    
    # Instantiating WordNetLemmatizer and lemmatizing the tokens
    lemmatizer = nltk.WordNetLemmatizer()
    clean_tokens = []
    
    for tok in tokens:
        if tok not in stopwords.words('english'):
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    """
    Builds a ML model to process text messages and classify them using DecisionTreeClassifier.

    Output:
        model --> The built model
    """

    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(DecisionTreeClassifier())),
    ])

    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (.8, 1.15)                  
    }

    # Creating the model using GridSearchCV
    model = GridSearchCV(estimator=pipeline, param_grid=parameters, verbose=6)
    
    return model

def train(X, y, model, categories):
    """
    Train the `model` using `X` and `y` and report the results
    
    Arguments:
        model --> sklearn model
        X_test --> Features for the test set
        Y_test --> Labels for the test set
        categories --> Possible categories for the messages
    """

    # Training test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Fitting model
    print('\n\nFitting Model\n\n')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=categories)

    # Printing model results
    for cat in categories:
        print('------------------------------------------------------\n')
        print(f'Feature: {cat}\n')
        print(classification_report(y_test[cat],y_pred_df[cat]))

    return model
    
def export_model(model, export_path):
    """
    Exports a trained model as a Pickle file.
    
    Arguments:
        model --> sklearn pipeline object
        export_path --> Destination path to save the `model`.
    """

    pickle.dump(model, open(export_path, 'wb'))
    
    pass

def run_pipeline(db_path, model_path):
    """
    Executes a pipeline, extracting data from a SQLite database,
    training a ML model with the data extracted, estimating its
    performance on the test set and then exporting it as a pickle file.
    
    Arguments:
        db_path --> Path of the SQLite database
        model_path --> Path to export the trained model
    """
    
    # Loading data from the SQLite database
    X, y, categories = load_data(db_path, db_path.replace('_db.db','') + '_tbl')  
    
    # Building the model
    model = build_model()  
    
    # Training the model
    model = train(X, y, model, categories)
    
    # Exporting the model as a pickle file
    export_model(model, model_path)


if __name__ == '__main__':
    # Getting paths for the SQLite database and the model from CLI arguments
    db_path = sys.argv[1]  
    model_path = sys.argv[2]  
    
    # Running the data pipeline
    run_pipeline(db_path, model_path)  