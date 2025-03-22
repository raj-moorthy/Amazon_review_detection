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
@st.cache_data
def load_data():
    file_path = 'C:/Users/rajre/OneDrive/Documents/Downloads/amazon.csv'  # Ensure this file is in the same directory or provide the correct path
    df = pd.read_csv(file_path)
    
    # Ensure 'reviewText' and 'score_pos_neg_diff' exist in the dataset
    if 'reviewText' not in df.columns or 'score_pos_neg_diff' not in df.columns:
        st.error("Dataset is missing required columns: 'reviewText' and 'score_pos_neg_diff'")
        return None
    
    df.dropna(subset=['reviewText', 'score_pos_neg_diff'], inplace=True)
    
    label_encoder = LabelEncoder()
    df['score_pos_neg_diff_encoded'] = label_encoder.fit_transform(df['score_pos_neg_diff'])
    
    scaler = StandardScaler()
    df['score_pos_neg_diff_standardized'] = scaler.fit_transform(df[['score_pos_neg_diff']])
    
    return df

@st.cache_resource
def train_model(df):
    x = df['reviewText']
    y_binary = [1 if score > 0 else 0 for score in df['score_pos_neg_diff']]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y_binary, test_size=0.2, random_state=42)
    
    # Vectorize text
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    
    # Train model
    model = LogisticRegression()
    model.fit(x_train_tfidf, y_train)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, model.predict(x_test_tfidf))
    
    return model, tfidf_vectorizer, accuracy

# Function to predict sentiment
def predict_sentiment(model, vectorizer, review):
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    return 'Positive' if prediction == 1 else 'Negative'

# Load Data and Train Model
df = load_data()
if df is not None:
    model, vectorizer, accuracy = train_model(df)

    # Streamlit UI
    st.title("Amazon Review Sentiment Analysis")
    st.write(f"Model Accuracy: {accuracy:.2f}")

    review_input = st.text_area("Enter a product review:")
    
    if st.button("Predict Sentiment"):
        sentiment = predict_sentiment(model, vectorizer, review_input)
        st.write(f"Sentiment: {sentiment}")
