import streamlit as st
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
def load_data():
    df = pd.read_csv('C:/Users/rajre/OneDrive/Documents/Downloads/amazon.csv')
    df.dropna(subset=['reviewText'], inplace=True)
    df['score_pos_neg_diff_encoded'] = LabelEncoder().fit_transform(df['score_pos_neg_diff'])
    scaler = StandardScaler()
    df['score_pos_neg_diff_standardized'] = scaler.fit_transform(df[['score_pos_neg_diff']])
    return df

def train_model(df):
    x = df['reviewText']
    y_binary = [1 if score > 0 else 0 for score in df['score_pos_neg_diff']]
    x_train, x_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    
    model = LogisticRegression()
    model.fit(x_train_tfidf, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test_tfidf))
    
    return model, tfidf_vectorizer, accuracy

def predict_sentiment(model, vectorizer, review):
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    return 'Positive' if prediction == 0 and 1 else 'Negative'

# Load Data and Train Model
df = load_data()
model, vectorizer, accuracy = train_model(df)

# Streamlit UI
st.title("Amazon Review Sentiment Analysis")

review_input = st.text_area("Enter a product review:")
if st.button("Predict Sentiment"):
    sentiment = predict_sentiment(model, vectorizer, review_input)
    st.write(f"Sentiment: {sentiment}")
