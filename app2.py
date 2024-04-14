import os
import string
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from utils.b2 import B2
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


REMOTE_DATA = "cnn_newsubset.csv.csv"

load_dotenv()


# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
         key_id=os.environ['B2_keyID'],
         secret_key=os.environ['B2_applicationKey'])


def get_data():
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df = b2.get_df(REMOTE_DATA)

    return df

# Retrieve the DataFrame
df = get_data()

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Perform stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Join the tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to extract text content from CNN article URL
def extract_text_from_url(article_url):
    response = requests.get(article_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = ' '.join([paragraph.text for paragraph in paragraphs])
    return text_content

# Streamlit app
def main():
    st.title('CNN Article Summarizer')

    # User input for article link
    article_url = st.text_input('Enter the URL of a CNN article')

    if st.button('Generate Summary'):
        try:
            # Extract text content from the article URL
            article_text = extract_text_from_url(article_url)

            # Preprocess the article text
            preprocessed_text = preprocess_text(article_text)

            # Initialize the summarization pipeline
            summarizer = pipeline("summarization")

            # Generate summary
            summary = summarizer(preprocessed_text, max_length=150, min_length=30, do_sample=False)

            # Display the summary in bullet points
            st.write('**Summary:**')
            for sentence in summary[0]['summary_text'].split('.'):
                st.write(f'- {sentence.strip()}')

        except Exception as e:
            st.error(f"An error occurred: {e}")
