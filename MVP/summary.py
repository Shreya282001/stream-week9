import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset into a DataFrame
import pandas as pd

dataset_path = r'c:\Users\SHREYA\OneDrive\Documents\MVP\WASHINGTON (CNN).xlsx'
df = pd.read_excel(dataset_path)

def preprocess_text(article_text):
    # Tokenize into sentences
    sentences = sent_tokenize(article_text)
    
    # Remove HTML tags if any
    sentences = [re.sub('<.*?>', '', sentence) for sentence in sentences]
    
    # Remove special characters and lowercase
    sentences = [re.sub(r'[^a-zA-Z0-9\s]', '', sentence).lower() for sentence in sentences]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    sentences = [' '.join(word for word in sentence.split() if word not in stop_words) for sentence in sentences]
    
    return sentences

# Preprocess each article in the dataset
preprocessed_articles = []
for article_text in df['article']:
    preprocessed_articles.append(preprocess_text(article_text))

# Display preprocessed articles
for i, article_sentences in enumerate(preprocessed_articles[:5]):  # Displaying only the first 5 articles
    print(f"Article {i+1}:")
    for sentence in article_sentences:
        print(sentence)
    print("------------------------")

# Join preprocessed sentences back into articles
preprocessed_texts = [' '.join(sentences) for sentences in preprocessed_articles]

# Calculate TF-IDF scores
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Define the number of sentences for the summary
num_sentences_summary = 3  # Adjust based on desired summary length

# Generate summaries
summaries = []
for i, article_sentences in enumerate(preprocessed_articles):
    # Calculate the sum of TF-IDF scores for each sentence
    sentence_scores = [sum(tfidf_matrix[i, vectorizer.vocabulary_[word]] for word in sentence.split() if word in vectorizer.vocabulary_) for sentence in article_sentences]
    
    # Sort the sentences based on their scores and select the top sentences
    top_sentence_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences_summary]
    
    # Select the top sentences and join them to form the summary
    summary_sentences = [article_sentences[j] for j in top_sentence_indices]
    summary = ' '.join(summary_sentences)
    
    # Append the summary to the list of summaries
    summaries.append(summary)

# Save summaries to a new Excel file
output_path = r'c:\Users\SHREYA\OneDrive\Documents\MVP\summaries.xlsx'  # Adjust the output path as needed
summary_df = pd.DataFrame({'Summary': summaries})
summary_df.to_excel(output_path, index=False)

print("Summaries saved to:", output_path)