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