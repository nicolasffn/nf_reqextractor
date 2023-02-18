import pandas as pd
from process_text import process_text
import utilities
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

if __name__ == "__main__":
    # load the required NLTK data and resources
    utilities.download("punkt")
    utilities.download('universal_tagset')

    # read the requirements.csv file into a dataframe
    df = pd.read_csv('./assets/csv/requirements.csv')

    # get the sentences and labels from the dataframe
    sentences = df['text'].tolist()
    labels = df['label'].tolist()

    # use the list of sentences to call the process_text function 
    # and write the result to a csv file
    df_results = pd.DataFrame(columns=['label', 'sentence', 'syntax'])
    for i, sentence in enumerate(sentences):
        print(i)
        entities = process_text(sentence)
        syntax = ''
        for entity in entities:
            syntax = syntax + entity[0] + ' '
        df_temp = pd.DataFrame({'label': [labels[i]], 'sentence': [sentence], 'syntax': [syntax]})
        df_results = pd.concat([df_results, df_temp], ignore_index=True)
    df_results.to_csv('./assets/csv/results.csv', index=False)

    # Read the CSV file into a dataframe
    df = pd.read_csv('./assets/csv/results.csv')

    # Split the data into training and testing sets
    req_train, req_test = train_test_split(df, test_size=0.2)

    # Define the pipeline for the RandomForestClassifier model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier())
    ])

    # Fit the model on the training data
    pipeline.fit(req_train['syntax'], req_train['label'])

    # Evaluate the model on the test data
    y_pred = pipeline.predict(req_test['syntax'])

    acc = accuracy_score(req_test['label'], y_pred)
    prec = precision_score(req_test['label'], y_pred, average=None)
    mc = confusion_matrix(y_pred, req_test['label'])

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Confusion matrix:\n", mc)