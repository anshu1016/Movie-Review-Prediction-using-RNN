import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB Dataset
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Helper Functions
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit UI
st.title("IMDB Movie Review Sentiment Analysis")
st.subheader("Enter a movie review to classify it as Positive or Negative")

# User input
user_input = st.text_area("Movie Review")
if st.button("Classify"):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)
        sentiment, score = predict_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.2f}")
    else:
        st.write("Please write a movie review.")
