# NF_REQEXTRACTOR
Software Requirement Classification

## Description
This project is a text classification model to classify software requirements into different categories.
The model is trained on a dataset of software requirements with labels, and then predicts the label of a new requirement based on its text. The model uses a combination of tokenization, padding, and a neural network architecture consisting of an embedding layer, an LSTM layer, and a dense layer.

## Requirements
The following libraries must be installed to run this project:
- pandas
- numpy
- scikit-learn
- tensorflow
- keras

## Usage
The following code reads the CSV file of software requirements and preprocesses the data for training and testing the model. The data is split into training and testing sets, encoded as integers, tokenized, and padded to the same length. The model architecture is defined, compiled, and trained on the training data. The function predict_requirement takes a text as input and returns the predicted label.

## Results
The model was trained for 250 epochs and reached a binary accuracy of XX on the test data.
The results of the model on a set of sample inputs are provided below.

```python
phrases = [
"The software must be able to calculate the sum of two numbers.",
"The website must allow users to search for products.",
"The system must be available 99.99% of the time.",
"The database must have a backup system to ensure data availability in case of failure.",
"The software must be able to recover from a crash and resume normal operations without data loss.",
"The system must detect and handle network failures without affecting the user experience.",
"The software must comply with GDPR regulations.",
"The website must include a disclaimer and privacy policy.",
"The software must have a modern and user-friendly interface.",
"The website must use consistent branding and follow a clean design.",
"The code must be well-documented to facilitate future maintenance and upgrades.",
"The system must have a monitoring system to detect potential issues before they become critical.",
"The software must be easy to install and configure.",
"The system must provide detailed logs..."
]

for phrase in phrases:
    print(f"Input: {phrase}\nPrediction: {predict_requirement(phrase)}")
```

## Explaination

Part 1: Imports the required libraries and modules.
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
```

Part 2: Loads the data from a CSV file, splits it into training and testing sets, and encodes the labels as integers.
```python
# Read the CSV file into a dataframe
df = pd.read_csv('./assets/csv/requirements.csv')

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Encode the labels as integers
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])
```

1. `df = pd.read_csv('./assets/csv/requirements.csv')`: Read the CSV file into a Pandas dataframe.
2. `train_df, test_df = train_test_split(df, test_size=0.2)`: Split the data into training and testing sets, with 80% of the data for training and 20% for testing.
3. `label_encoder = LabelEncoder(); train_df['label'] = label_encoder.fit_transform(train_df['label'])`: Encode the labels as integers, by creating a `LabelEncoder` object and fitting it to the training labels.
4. `test_df['label'] = label_encoder.transform(test_df['label'])`: Transform the testing labels using the same `LabelEncoder` object.

Part 3: Tokenizes the text data and pads the sequences to the same length. The labels are also converted to one-hot encoded arrays.
```# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_df['text'])
train_sequences = tokenizer.texts_to_sequences(train_df['text'])
test_sequences = tokenizer.texts_to_sequences(test_df['text'])

# Pad the sequences to the same length
max_length = max([len(s) for s in train_sequences + test_sequences])
train_data = pad_sequences(train_sequences, maxlen=max_length)
test_data = pad_sequences(test_sequences, maxlen=max_length)

# Convert the labels to one-hot encoded arrays
train_labels = to_categorical(train_df['label'], num_classes=len(label_encoder.classes_))
test_labels = to_categorical(test_df['label'], num_classes=len(label_encoder.classes_))
```

1. `tokenizer = Tokenizer(num_words=5000)`: Initialize a `Tokenizer` object with a vocabulary of 5000 words. The `Tokenizer` class is used to tokenize the text into numerical sequences, which can be understood by the machine learning models. 
2. `tokenizer.fit_on_texts(train_df['text'])`: Train the tokenizer on the training text data.
3. `train_sequences = tokenizer.texts_to_sequences(train_df['text'])`: Convert the training text data into numerical sequences.
4. `test_sequences = tokenizer.texts_to_sequences(test_df['text'])`: Convert the testing text data into numerical sequences.
5. `max_length = max([len(s) for s in train_sequences + test_sequences])`: Calculate the maximum length of the sequences, which will be used to pad the sequences to the same length later.
6. `train_data = pad_sequences(train_sequences, maxlen=max_length)`: Pad the training sequences to the same length.
7. `test_data = pad_sequences(test_sequences, maxlen=max_length)`: Pad the testing sequences to the same length.
8. `train_labels = to_categorical(train_df['label'], num_classes=len(label_encoder.classes_))`: Convert the training labels into one-hot encoded arrays. The `to_categorical` function is from the `keras.utils.np_utils` module.
9. `test_labels = to_categorical(test_df['label'], num_classes=len(label_encoder.classes_))`: Convert the testing labels into one-hot encoded arrays.

Part 4: Defines the architecture of the neural network and trains it on the training data.
```# Define the neural network architecture
model = Sequential()
model.add(Embedding(5000, 100, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=150, batch_size=600, validation_data=(test_data, test_labels))
```

1. `model = Sequential()`: Initializes a new neural network model using the `Sequential` class from the `tensorflow.keras.models module`. This creates an empty linear stack of layers.
2. `model.add(Embedding(5000, 100, input_length=max_length))`: Adds an embedding layer to the model with 5000 output dimensions, a 100 dimensional input and a maximum input length of `max_length`.
3. `model.add(SpatialDropout1D(0.2))`: Adds a dropout layer with a rate of 20% to prevent overfitting.
4. `model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))`: Adds a long short-term memory layer with 256 units and a dropout rate of 20% for both the input and recurrent connections to prevent overfitting.
5. `model.add(Dense(len(label_encoder.classes_), activation='softmax'))`: Adds a dense layer with an output size equal to the number of categories and a `softmax` activation function.
6. `model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['binary_accuracy'])`: Compiles the model by specifying the loss function `categorical_crossentropy` which is suitable for multi-class classification problems, the optimizer `adam` which is an efficient gradient-based optimization algorithm and the evaluation metric `binary_accuracy`.
7. `model.fit(train_data, train_labels, epochs=250, batch_size=600, validation_data=(test_data, test_labels))`: Trains the model on the training data `train_data` and corresponding labels `train_labels` for 250 epochs with a batch size of 600 and evaluating the model on the test data `test_data` and labels `test_labels`.

Part 5: Defines a function to predict the sentiment of an input text using the trained model. The input text is tokenized and padded, then the prediction is made based on the model. The label is then decoded and the confidence of the prediction is also returned.
```python
def predict_requirement(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    data = pad_sequences(sequences, maxlen=max_length)
    # Predict the label
    prediction = model.predict(data)
    # Decode the label
    label_index = prediction.argmax()
    label = label_encoder.inverse_transform([label_index])
    confidence = prediction[0][label_index]
    return label[0], confidence
```

1. `sequences = tokenizer.texts_to_sequences([text])`: Calls the `texts_to_sequences` function of the `tokenizer` object, passing in a list containing the text argument. This will convert the text into a sequence of integers representing the words in the text.
2. `data = pad_sequences(sequences, maxlen=max_length)`: Calls the `pad_sequences` function to pad the `sequences` array to a length of `max_length`. This will ensure that all sequences have the same length and can be processed by the neural network.
3. `prediction = model.predict(data)`: Calls the `predict` function of the `model` object, passing in the `data` array. This will run a prediction for the input `text` and return a 2D array of probabilities for each label, representing the model's confidence in each label.
4. `label_index = prediction.argmax()`: Calls the `argmax` function on the `prediction` array to find the index of the maximum value, which represents the predicted label.
5. `label = label_encoder.inverse_transform([label_index])`: Calls the `inverse_transform` function of the `label_encoder` object, passing in a list containing the `label_index`. This will convert the integer index back into the original label string.
6. `confidence = prediction[0][label_index]`: Accesses the element at position `[0][label_index]` in the `prediction` array to retrieve the confidence score for the predicted label.
