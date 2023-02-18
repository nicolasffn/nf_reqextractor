import pandas as pd
from process_text import process_text
import utilities

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support, precision_score, recall_score

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
    train_df, test_df = train_test_split(df, test_size=0.2)

    # Encode the labels as integers
    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['label'])
    test_df['label'] = label_encoder.transform(test_df['label'])

    # Tokenize the text
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_df['syntax'])
    train_sequences = tokenizer.texts_to_sequences(train_df['syntax'])
    test_sequences = tokenizer.texts_to_sequences(test_df['syntax'])

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
    model.fit(train_data, train_labels, epochs=250, batch_size=535, validation_data=(test_data, test_labels))

    # Make predictions on the test set and calculate metrics
    y_true = np.argmax(test_labels, axis=1)
    y_pred = np.argmax(model.predict(test_data), axis=1)
    accuracy = accuracy_score(y_true, y_pred, zero_division=1)
    print(f"Accuracy: {accuracy:.2f}")

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    # Print precision, recall, and f1 score for each label
    for label, prec, rec, f1_score, support_count in zip(label_encoder.classes_, precision, recall, f1, support):
        print(f"Label: {label}")
        print(f"  Precision: {prec:.2f}")
        print(f"  Recall: {rec:.2f}")
        print(f"  F1 score: {f1_score:.2f}")
        print(f"  Support: {support_count}")

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
    # Decode the label
    label_index = prediction.argmax()
    label = label_encoder.inverse_transform([label_index])
    confidence = prediction[0][label_index]
    return label[0], confidence

phrases = ["The software must be able to calculate the sum of two numbers.",
"The website must allow users to search for products.",
"The system must be available 99.99% of the time.",
"The database must have a backup system to ensure data availability in case of failure.",
"The software must comply with GDPR regulations.",
"The website must include a disclaimer and privacy policy.",
"The software must have a modern and user-friendly interface.",
"The website must use consistent branding and follow a clean design.",
"The software must be able to process 1000 requests per second.",
"The website must load quickly and efficiently, even under heavy traffic.",
"The software must be able to handle an increase in users and data without significant performance degradation.",
"The system must be able to add additional resources, such as servers, as needed to maintain performance.",
"The software must use secure protocols for transmitting data.",
"The system must have strict access controls and user authentication mechanisms in place.",
"The software must have clear and intuitive navigation.",
"The website must be accessible for users with disabilities."]

for phrase in phrases:
    entities = process_text(phrase)
    syntax = ''
    for entity in entities:
        syntax = syntax + entity[0] + ' '
    result = predict_requirement(syntax)
    print(f"Phrase: {phrase}\nInput: {syntax}\nPrediction: {result[0]}\nConfidence: {result[1]}")