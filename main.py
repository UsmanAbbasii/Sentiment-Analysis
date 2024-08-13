import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# Load the IMDB dataset word index from the JSON file
with open('word_index.json', 'r') as f:
    word_index = json.load(f)

reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Function to generate a word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate(text)
    return wordcloud

# Streamlit app
st.set_page_config(page_title="Sentiment Analysis", page_icon="🔍")

# Title and introduction
st.title('Sentiment Analysis 🔍')

# Custom styling for dark mode
st.markdown("""
    <style>
        .stTextInput>div>div>textarea {
            border: 2px solid #1e1e1e;
            border-radius: 8px;
            background-color: #333;
            color: #ddd;
            padding: 10px;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .info-section {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            color: #ddd;
        }
        .footer {
            text-align: center;
            padding: 20px;
            background-color: #1e1e1e;
            border-top: 1px solid #333;
            color: #ddd;
        }
    </style>
    """, unsafe_allow_html=True)

# Information about the project
st.markdown("""
    <div class="info-section">
        <h3>About This Project</h3>
        <p>
            This application performs sentiment analysis using a Simple RNN model. While this model is effective for basic sentiment classification tasks, there are more advanced architectures and models available for sentiment analysis. 
            <br><br>
            <strong>Why Simple RNN?</strong> 
            The focus here is to demonstrate the functionality and implementation of a Simple RNN model. It serves as a foundational approach to understand sentiment analysis before delving into more complex models like LSTMs, GRUs, or Transformer-based architectures.
            <br><br>
            Stay tuned for future updates where I will explore and apply advanced models to further enhance sentiment analysis capabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)

# User input
user_input = st.text_area('Enter your review:', height=150)

if st.button('Classify Review'):

    if user_input:
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result with emojis
        sentiment_emoji = '😊' if sentiment == 'Positive' else '😞'
        st.write(f'**Sentiment:** {sentiment} {sentiment_emoji}')
        st.write(f'**Prediction Score:** {prediction[0][0]:.4f}')

        # Generate and display word cloud
        wordcloud = generate_wordcloud(user_input)
        st.subheader('Word Cloud of Your Review')
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        # Save word cloud to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        st.image(buf, use_column_width=True)
        plt.close()

    else:
        st.error('Please enter a review to classify.')

# Footer with styling
st.markdown("""
    <div class="footer">
        Made with ❤️ using Streamlit and TensorFlow.
    </div>
    """, unsafe_allow_html=True)
