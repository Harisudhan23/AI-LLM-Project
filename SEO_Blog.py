import streamlit as st
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import spacy
import logging

# Constants
API_KEY = "AIzaSyDR0Sr1VV1TJl3AFNScIdubB7JkyUsJhSo"  # Replace with your actual API key
LLM_MODEL = "gemini-1.5-pro-002"

# Initialize LLM
llm = ChatGoogleGenerativeAI(api_key=API_KEY, model=LLM_MODEL)

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Configure logging
logging.basicConfig(level=logging.ERROR, filename="error_log.txt")


# Utility Functions
def log_error(message, error):
    """Logs error messages to a file."""
    logging.error(f"{message}: {error}")


def retrieve_blog_content(url):
    """Fetches and parses blog content from a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        content = " ".join([p.text for p in soup.find_all('p')])
        title = soup.title.string.strip() if soup.title else "No title found"
        meta_tag = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
        meta_description = meta_tag["content"].strip() if meta_tag else "No meta description found"

        if not content.strip():
            raise ValueError("Blog content is empty or could not be retrieved.")

        return content, title, meta_description
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving blog content: {e}")
        log_error("RequestException in retrieve_blog_content", e)
    except ValueError as e:
        st.error(f"Content Error: {e}")
        log_error("ValueError in retrieve_blog_content", e)
    return None, None, None


def extract_keywords_from_content(content):
    """Extracts keywords from blog content using NLP."""
    try:
        doc = nlp(content)
        keywords = [chunk.text.lower() for chunk in doc.noun_chunks]
        keywords = list(set(keywords))  # Remove duplicates
        return keywords
    except Exception as e:
        log_error("Error in extract_keywords_from_content", e)
        return []


def optimize_seo_keywords(content, checklist_keywords):
    """Optimizes SEO based on keyword checklist criteria and content analysis."""
    try:
        prompt = PromptTemplate(
            input_variables=["content", "checklist_keywords"],
            template="""Analyze the following blog content and evaluate SEO based on the provided checklist keywords:
            
            Blog Content:
            {content}
            
            Checklist Keywords:
            {checklist_keywords}

            Please perform the following checks:
            1. Check if the target keyword is relevant and aligned with user search intent.
            2. Confirm if the keyword is not overused or duplicated across other pages.
            3. Verify if the page title contains the primary keyword and is click-worthy.
            4. Suggest modifiers that can enhance the title's SEO potential.
            5. Check the length of the title and if it’s optimized.
            6. Ensure that the primary keyword is in the H1 tag and meta description.
            7. Ensure that the URL is SEO-friendly and contains the primary keyword.
            8. Confirm that the primary keyword is included in the first sentence.
            9. Review the keyword density and ensure it’s not over-optimized.
            10. Check if keyword variations and LSI keywords are integrated into the copy.

            Provide actionable suggestions for improvement in a list format.
            """
        )

        checklist_keywords_str = ", ".join(checklist_keywords)
        response = llm.invoke({
            "content": content,
            "checklist_keywords": checklist_keywords_str,
        })

        return response.content.strip().split("\n")
    except Exception as e:
        log_error("Error in optimize_seo_keywords", e)
        return ["An error occurred during SEO optimization."]


def visualize_word_cloud(keywords):
    """Creates a word cloud visualization for the given keywords."""
    if not keywords:
        st.warning("No keywords to display.")
        return
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(keywords))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
        log_error("Error in visualize_word_cloud", e)


# Streamlit App
def main():
    st.title("Blog SEO Analyzer")

    # Input blog URL
    blog_url = st.text_input("Enter Blog URL:")
    if st.button("Analyze Blog"):
        with st.spinner("Analyzing blog..."):
            # Retrieve blog content
            content, title, meta_description = retrieve_blog_content(blog_url)
            if not content:
                return

            # Extract keywords from blog content
            extracted_keywords = extract_keywords_from_content(content)
            st.subheader("Extracted Keywords from Blog Content")
            st.write(", ".join(extracted_keywords))

            # Optimize SEO using extracted keywords
            seo_suggestions = optimize_seo_keywords(content, extracted_keywords)

            # Display results
            st.subheader("SEO Optimization Suggestions")
            for suggestion in seo_suggestions:
                st.write(f"- {suggestion}")

            st.subheader("Keyword Word Cloud")
            visualize_word_cloud(extracted_keywords)


if __name__ == "__main__":
    main()
