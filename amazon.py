import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer
with open("C:/Users/rajre/OneDrive/Documents/Downloads/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("C:/Users/rajre/OneDrive/Documents/Downloads/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_sentiment(review):
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    return "Positive" if prediction == 0 and 1 else "Negative"

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a product review to analyze its sentiment:")

review_input = st.text_area("Review Text")
if st.button("Predict Sentiment"):
    if review_input.strip():
        sentiment = predict_sentiment(review_input)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a review text.")