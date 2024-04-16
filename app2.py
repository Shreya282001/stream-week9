import streamlit as st
from txtai.pipeline import Summary
from bs4 import BeautifulSoup
import requests

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def text_summary(text, maxlength=None):
    # Create summary instance
    summary = Summary()
    text = (text)
    result = summary(text)
    return result

def extract_text_from_url(url):
    # Send a GET request to the URL
    response = requests.get(url)
    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")
    # Find all paragraph elements
    paragraphs = soup.find_all("p")
    # Concatenate paragraphs into a single text
    text = "\n".join([p.get_text() for p in paragraphs])
    return text

choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document", "Summarize URL"])

if choice == "Summarize Text":
    st.subheader("Summarize Text using txtai")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                result = text_summary(input_text)
                st.success(result)

elif choice == "Summarize Document":
    st.subheader("Summarize Document using txtai")
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1,1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                st.markdown("**Extracted Text is Below:**")
                st.info(extracted_text)
            with col2:
                st.markdown("**Summary Result**")
                text = extract_text_from_pdf("doc_file.pdf")
                doc_summary = text_summary(text)
                st.success(doc_summary)

elif choice == "Summarize URL":
    st.subheader("Summarize Article from URL using txtai")
    article_url = st.text_input("Enter the URL of the article")
    if article_url:
        if st.button("Summarize URL"):
            try:
                article_text = extract_text_from_url(article_url)
                st.markdown("**Extracted Text from the URL:**")
                st.info(article_text)
                st.markdown("**Summary Result:**")
                summary = text_summary(article_text)
                # Split summary into bullet points
                summary_bullet_points = summary.split("\n")
                for point in summary_bullet_points:
                    st.write(f"- {point}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
