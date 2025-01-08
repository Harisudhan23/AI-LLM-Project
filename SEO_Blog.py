import streamlit as st
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import spacy
import logging
import textstat

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

def calculate_readability(content):
    """Calculates Flesch-Kincaid grade and reading ease for content."""
    try:
        if not content or not content.strip():
            return None, None
        kincaid_grade = textstat.flesch_kincaid_grade(content)
        reading_ease = textstat.flesch_reading_ease(content)
        return kincaid_grade, reading_ease
    except Exception as e:
        log_error("Error in calculate_readability", e)
        return None, None

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


def optimize_seo_keywords(content, page_title, meta_description, url):
    """Optimizes SEO keywords based on structured guidelines."""
    try:
        prompt = PromptTemplate(
            input_variables=["content", "page_title", "meta_description", "url"],
            template="""Analyze the following content and evaluate it based on the SEO keyword optimization guidelines. Provide specific suggestions for improvement where applicable:

            Content:
            {content}

            Page Title:
            {page_title}

            Meta Description:
            {meta_description}

            URL:
            {url}

            SEO Keyword Optimization Guidelines:
            1. Ensure you are targeting the right keyword that aligns with the content and search intent.
            2. Verify that the page satisfies the search intent of the target audience for the given keyword.
            3. Confirm that the primary keyword is included in the page title.
            4. Assess if the title is engaging and click-worthy to improve click-through rates.
            5. Check if there are opportunities to add modifiers to the title (e.g., "Best", "Top", "Guide", "2025").
            6. Ensure that you have utilized all available space in the title tag, without exceeding character limits.
            7. Verify that the page title is wrapped in an H1 tag to ensure proper HTML structure.
            8. Check if the primary keyword is included in the meta description.
            9. Confirm that the primary keyword is present in the URL to improve search relevance.
            10. Ensure the URL structure is concise, descriptive, and SEO-friendly, avoiding unnecessary parameters.
            11. Ensure that the primary keyword is included in the first sentence of the content for better SEO alignment.
            12. Assess whether the keyword density is optimized and not too aggressive compared to competitor content.
            13. Ensure that variations of the primary keyword are included throughout the content to improve relevance.
            14. Check if synonyms or LSI (Latent Semantic Indexing) keywords are used in the copy to enhance content diversity and SEO.

            Based on the analysis, provide an evaluation of how well the content adheres to these guidelines and any actions that should be taken to improve SEO performance.
            """
        )

        response = (prompt | llm).invoke({
            "content": content,
            "page_title": page_title,
            "meta_description": meta_description,
            "url": url,
        })
        
        # Process and return the detailed evaluation as a list
        return response.content.strip().split("\n")
    except Exception as e:
        st.error(f"Error optimizing SEO keywords: {e}")
        log_error("Error in optimize_seo_keywords", e)
        return []


def evaluate_content_quality(content):
    """Evaluates content quality based on structured guidelines using an LLM."""
    try:
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""Analyze the following content and evaluate it based on the guidelines below. Provide specific suggestions for improvement where applicable:

            Blog Content:
            {content}

            Content Quality Guidelines:
            1. Ensure the copy is free of spelling and grammatical errors. If any issues are identified, suggest corrections.
            2. The content should be scannable. Ensure it is easy to read with proper use of headings, bullet points, and concise formatting.
            3. The content should be written at a readability level suitable for an 8th grader. Check for overly complex language or sentence structures.
            4. Ensure the copy is engaging, holding the reader’s attention throughout. Suggest improvements to increase engagement if necessary.
            5. Use short paragraphs to improve readability and avoid dense blocks of text.
            6. The headings should be structured logically, guiding the reader through the content in a coherent flow.
            7. Ensure that the headings are descriptive and clearly indicate the topic of each section.
            8. Use keyword variations, LSI keywords, or synonyms in the headings to improve SEO and make the content more varied.
            9. Incorporate bullet points and numbered lists where applicable to enhance clarity and structure.
            10. Ensure the copy is “fresh” by being original, up-to-date, and relevant to current trends or information.

            Provide an evaluation based on these guidelines, along with any suggestions for improvement where necessary. Confirm if the content meets the guidelines or provide specific actions for optimization.
            """
        )

        response = (prompt | llm).invoke({
            "content": content,
        })
        
        # Process and return the detailed evaluation as a list
        return response.content.strip().split("\n")
    except Exception as e:
        st.error(f"Error evaluating content quality: {e}")
        log_error("Error in evaluate_content_quality", e)
        return []




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

            readability = calculate_readability(content)
            st.subheader("Readability Scores")
            st.metric("Flesch-Kincaid Grade", f"{readability[0]:.2f}" if readability[0] else "N/A")
            st.metric("Flesch Reading Ease", f"{readability[1]:.2f}" if readability[1] else "N/A")  
            
            # Extract keywords from blog content
            extracted_keywords = extract_keywords_from_content(content)
            st.subheader("Extracted Keywords from Blog Content")
            st.write(", ".join(extracted_keywords))

            # Optimize SEO using extracted keywords
            st.subheader("SEO Suggestions:")
            seo_suggestions = optimize_seo_keywords(content, title, meta_description, blog_url)
            for suggestion in seo_suggestions:
                st.write(f"- {suggestion}")
            # Evaluate the content quality
            st.subheader("Content Evaluation:")
            content_suggestion = evaluate_content_quality(content)
            for suggestion in content_suggestion:
                st.write(f"- {suggestion}")


if __name__ == "__main__":
    main()
