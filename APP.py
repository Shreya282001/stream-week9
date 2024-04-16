import os
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from utils.b2 import B2
REMOTE_DATA = "cnn_newsubset.csv.csv"

load_dotenv()


# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
         key_id=os.environ['B2_keyID'],
         secret_key=os.environ['B2_applicationKey'])


@st.cache_data
def get_data():
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df = b2.get_df(REMOTE_DATA)
    return df

# Retrieve the DataFrame
df = get_data()

st.title("The length of the articles correlate with the length of the highlights?")
# Calculate the length of the article and the highlight
df['article_length'] = df['article'].str.len()
df['highlight_length'] = df['highlights'].str.len()

# Plot the data
fig = plt.figure(figsize=(10, 6))
plt.scatter(df['article_length'], df['highlight_length'], color='skyblue')
plt.xlabel('Length of Article')
plt.ylabel('Length of Highlight')
plt.title('Relationship between Article and Highlight Lengths')
plt.show()

st.pyplot(fig)
st.dataframe(df.head(25))