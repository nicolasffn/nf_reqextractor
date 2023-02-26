import pandas as pd
import utilities

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support, precision_score, recall_score

import pickle

# Download the required NLTK data and resources
utilities.download("punkt")
utilities.download('universal_tagset')

# Read the requirements.csv file into a dataframe
df = pd.read_csv('./assets/csv/requirements.csv')

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Encode the labels as integers
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

# Tokenize the text
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

use_saved_model = False
# Define the neural network architecture
if use_saved_model:
        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)
else:
    model = Sequential()
    model.add(Embedding(5000, 100, input_length=max_length)) # Word Embedding layer
    model.add(SpatialDropout1D(0.2)) # Spatial Dropout layer to prevent overfitting
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2)) # LSTM layer for sequence processing
    model.add(Dense(len(label_encoder.classes_), activation='softmax')) # Output layer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    # Train the model
    batch_size = 1000 # Number of samples per batch
    epochs = 250 # Number of times to iterate over the training data
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Save the trained model
    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)

# Make predictions on the test set and calculate metrics
y_true = np.argmax(test_labels, axis=1) # Convert one-hot encoded labels to integers
y_pred = np.argmax(model.predict(test_data), axis=1) # Make predictions and convert to integers
accuracy = accuracy_score(y_true, y_pred) # Calculate accuracy
print(f"Accuracy: {accuracy:.2f}")

# Calculate precision, recall, and f1 score for each label
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

# Print precision, recall, and f1 score for each label
for label, prec, rec, f1_score, support_count in zip(label_encoder.classes_, precision, recall, f1, support):
    print(f"Label: {label}")
    print(f" |--->Precision: {prec:.2f}")
    print(f" |--->Recall: {rec:.2f}")
    print(f" |--->F1 score: {f1_score:.2f}")
    print(f" |--->Support: {support_count}")

# Calculate and print overall precision, recall, and f1 score
macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()
micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=1)
weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
# Calculate the precision and recall for each class
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted')
# Calculate the micro-averaged precision and recall
micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=1)
micro_recall = recall_score(y_true, y_pred, average='micro')

# Print overall precision, recall, and f1 score
print(f"Weighted precision: {precision:.2f}")
print(f"Weighted recall: {recall:.2f}")
print(f"Macro-average precision: {macro_precision:.2f}")
print(f"Macro-average recall: {macro_recall:.2f}")
print(f"Macro-average F1 score: {macro_f1:.2f}")
print(f"Micro-average precision: {micro_precision:.2f}")
print(f"Micro-average recall: {micro_recall:.2f}")

# Print the confusion matrix
confusion_matrix = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(confusion_matrix)

def predict_requirement(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    data = pad_sequences(sequences, maxlen=max_length)
    # Predict the label
    prediction = model.predict(data)
    # Get the indices of the top 3 confidence scores
    top_3_indices = np.argsort(prediction[0])[-3:][::-1]
    # Get the labels and confidence scores for the top 3 predictions
    top_3_labels = label_encoder.inverse_transform(top_3_indices)
    top_3_confidences = prediction[0][top_3_indices]
    return top_3_labels, top_3_confidences