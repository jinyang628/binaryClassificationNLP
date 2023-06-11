import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from nltk import WordNetLemmatizer
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import re
import string
import nltk
from collections import Counter
from textblob import TextBlob
import contractions



"""
Stop Words: A stop word is a commonly used word (such as "the", "a", "an", "in") that a search engine
has been programmed to ignore, both when indexing entries for searching and when retrieving them as 
the result of a search query.
"""
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words("english"))

df = pd.read_csv("dataset.csv")
# print(df.shape)
# print(df.head())
#target == 1 means disaster tweet and vice versa
# print((df.target == 1).sum())
# print((df.target == 0).sum())

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"", text)

def remove_punct(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

def lower_case(text):
    lower_case_words = [word.lower() for word in text.split()]
    return " ".join(lower_case_words)

def remove_contraction(text):
    return contractions.fix(text)

def remove_stopwords(text):
    filtered_words = [word for word in text.split() if word not in stop]
    return " ".join(filtered_words)

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"U0001F600-U0001F64F"  # emoticons
                           u"U0001F300-U0001F5FF"  # symbols & pictographs
                           u"U0001F680-U0001F6FF"  # transport & map symbols
                           u"U0001F1E0-U0001F1FF"  # flags (iOS)
                           u"U00002702-U000027B0"
                           u"U000024C2-U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# Computationally Intensive
def spell_check(text):
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return str(corrected_text)

# Computationally Intensive
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(lemmatized_words)

  # returns a dictionary where key is each unique word and value is the frequency count
def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


# Computationally intensive
# df["text"] = df.text.map(remove_URL).map(lower_case).map(remove_punct).map(spell_check).map(lemmatize_text).map(remove_contraction).map(remove_stopwords).map(remove_emoji)

df["text"] = df.text.map(remove_URL).map(lower_case).map(remove_punct).map(remove_contraction).map(remove_stopwords).map(remove_emoji)

counter = counter_word(df.text)
num_unique_words = len(counter)

# Split the dataset into training, validation and test sets
training_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(training_df, test_size=0.2, random_state=42)

# split text and labels
train_sentences = train_df.text.to_numpy()
train_labels = train_df.target.to_numpy()
val_sentences = val_df.text.to_numpy()
val_labels = val_df.target.to_numpy()
test_sentences = test_df.text.to_numpy()
test_labels = test_df.target.to_numpy()

tokenizer = Tokenizer(num_words=(num_unique_words + 1), oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)


max_length = max(max(len(seq) for seq in train_sequences),
                 max(len(seq) for seq in val_sequences))
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# takes in the token sequence and returns the original sentence
reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])
def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])

# If same output, everything is right so far
print(decode(train_sequences[10]))
print(train_sentences[10])

model = keras.models.Sequential()

"""
An embedding layer is a mapping of discrete categorical inputs (such as words) to dense vectors, capturing semantic relationships between the inputs in a lower-dimensional space.
An LSTM (Long Short-Term Memory) layer is a type of recurrent neural network layer that retains memory of past information and is particularly effective in capturing long-term dependencies in sequential data.
A dense layer, also known as a fully connected layer, is a neural network layer where each neuron is connected to every neuron in the previous layer, allowing for complex transformations and nonlinearities in the data.
"""

model.add(Embedding(num_unique_words, 32, input_length=max_length))
# dropout parameter of 0.1 specifies that a fraction of the input units (10%) will be randomly set to 0 during training to prevent overfitting.
model.add(LSTM(64, dropout=0.1))
# dense layer is often used as the output layer for binary classification tasks since it can provide a probability score between 0 and 1
# activation="sigmoid" parameter applies the sigmoid activation function, which squashes the output to a probability range, indicating the likelihood of the binary class (0 or 1) for the given input
model.add(Dense(1, activation="sigmoid"))
model.summary()

# BinaryCrossentropy used for binary classification
# from_logits=False because we had an activation function above, so our output from the final layer is already in probabilities and not logits
loss = keras.losses.BinaryCrossentropy(from_logits=False)

# learning rate is set to relatively low value of 0.001 so there is slower convergence (state where an algorithm has reached an "optimal" solution)
optim = keras.optimizers.Adam(learning_rate=0.001)

metrics=["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

model.fit(train_padded, train_labels, epochs=9, validation_data=(val_padded, val_labels), verbose=2)

predictions = model.predict(test_padded)
predictions= [1 if p > 0.5 else 0 for p in predictions]

print(val_sentences[10:20])
print(val_labels[10:20])
print(predictions[10:20])

# Customised input

def preprocess_input(text):
    # Computationally intensive
    # return remove_emoji(remove_stopwords(remove_contraction(lemmatize_text(spell_check(remove_punct(lower_case(remove_URL(text))))))))
    return remove_emoji(remove_stopwords(remove_contraction(remove_punct(lower_case(remove_URL(text))))))

def predict_input(text):
    preprocessed_text = preprocess_input(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)[0]
    if prediction > 0.5:
        return "Disaster"
    else:
        return "Non-Disaster"

input_text = "There is a fire in the building!" # Change this
prediction = predict_input(input_text)
print("Prediction:", prediction)
