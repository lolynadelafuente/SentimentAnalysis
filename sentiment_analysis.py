# commands to launch streamlit in the terminal
# .venv\Scripts\Activate.ps1
# streamlit run sentiment_analysis.py
# deactivate

import streamlit as st
import re
from transformers import pipeline

# Load models only once
@st.cache_resource
def load_models():
    classifier_emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, truncation=True, max_length=512)
    classifier_number = pipeline("text-classification", model="siebert/sentiment-roberta-large-english", top_k=None, truncation=True, max_length=512)
    return classifier_emotion, classifier_number

classifier_emotion, classifier_number = load_models()

# Preprocessing function
def preprocess_text(prompt):
    prompt = prompt.lower().strip()  # Convert to lowercase and remove extra spaces
    prompt = re.sub(r'[^\w\s]', '', prompt)  # Remove punctuation
    sentences = re.split(r'(?<=[.!?])\s+', prompt)  # Split into sentences
    return sentences

# Function to classify sentiment
def classify_sentiment(prompt):
    
    sentences = preprocess_text(prompt)
    results = []
    
    for sentence in sentences:
        emotion_result = classifier_emotion(sentence)
        sentiment_result = classifier_number(sentence)
        
        # Get the highest confidence label
        top_emotion = max(emotion_result[0], key=lambda x: x['score'])
        top_sentiment = max(sentiment_result[0], key=lambda x: x['score'])
        
        results.append(f"'{prompt}' → This is a **{top_sentiment['label']}** text with **{top_emotion['label']}** as the predominant emotion")
    return results

# Streamlit UI
st.title("Sentiment Classifier")
st.write("Enter a sentence to analyze its sentiment and predominant emotion.")

user_input = st.text_area("How do you feel today?", "")

if st.button("Classify"):
    if not user_input.strip():  # Check if input is empty or only whitespace
        st.warning("⚠️ Please enter some text before submitting!")
    else:
        results = classify_sentiment(user_input)
        for res in results:
            st.markdown(res)
