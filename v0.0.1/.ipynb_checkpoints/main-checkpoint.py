import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical

# Read the CSV file into a dataframe
df = pd.read_csv('./assets/csv/requirements.csv')

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Encode the labels as integers
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
print(tokenizer)
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

# Define the neural network architecture
model = Sequential()
model.add(Embedding(5000, 100, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=250, batch_size=600, validation_data=(test_data, test_labels))

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

phrases = ["The software must be able to calculate the sum of two numbers.",
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
"The system must provide detailed logs for troubleshooting and auditing purposes.",
"The software must be able to process 1000 requests per second.",
"The website must load quickly and efficiently, even under heavy traffic.",
"The software must be compatible with Windows, macOS, and Linux operating systems.",
"The system must be able to integrate with other tools and systems through APIs.",
"The software must be able to handle an increase in users and data without significant performance degradation.",
"The system must be able to add additional resources, such as servers, as needed to maintain performance.",
"The software must use secure protocols for transmitting data.",
"The system must have strict access controls and user authentication mechanisms in place.",
"The software must have clear and intuitive navigation.",
"The website must be accessible for users with disabilities."]

for phrase in phrases:
    result = predict_requirement(phrase)
    print(f"Input: {phrase}\nPrediction: {result[0]}\nConfidence: {result[1]}")