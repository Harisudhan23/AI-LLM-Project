import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import textstat
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import logging

# Constants
API_KEY = "AIzaSyDR0Sr1VV1TJl3AFNScIdubB7JkyUsJhSo"  # Replace with your actual API key
LLM_MODEL = "gemini-1.5-pro-002"
LLM_TEMPERATURE = 0.3

# Initialize LLM
llm = ChatGoogleGenerativeAI(api_key=API_KEY, model=LLM_MODEL, temperature=LLM_TEMPERATURE)

# # Configure logging
# logging.basicConfig(level=logging.ERROR, filename="error_log.txt")

# # Utility Functions
# def log_error(message, error):
#     """Logs error messages to a file."""
#     logging.error(f"{message}: {error}")

# Step 1: Retrieve Blog Content
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
        #log_error("RequestException in retrieve_blog_content", e)
    except ValueError as e:
        st.error(f"Content Error: {e}")
        #log_error("ValueError in retrieve_blog_content", e)
    return None, None, None

# Step 2: Extract Domain and Subdomain
def extract_domain_subdomain(url):
    """Extracts the domain and subdomain from a URL."""
    try:
        parsed_url = urlparse(url)
        domain_parts = parsed_url.netloc.split('.')
        domain = ".".join(domain_parts[-2:])
        subdomain = parsed_url.netloc if len(domain_parts) > 2 else "No subdomain"
        return domain, subdomain
    except Exception as e:
        #log_error("Error in extract_domain_subdomain", e)
        return "Invalid domain", "Invalid subdomain"

# Step 3: Calculate Readability Score
def calculate_readability(content):
    """Calculates Flesch-Kincaid grade and reading ease for content."""
    try:
        if not content or not content.strip():
            return None, None
        kincaid_grade = textstat.flesch_kincaid_grade(content)
        reading_ease = textstat.flesch_reading_ease(content)
        return kincaid_grade, reading_ease
    except Exception as e:
        #log_error("Error in calculate_readability", e)
        return None, None

# Step 4: Suggest SEO Keywords
def suggest_seo_keywords(content):
    """Generates SEO keyword suggestions using the LLM."""
    try:
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""Analyze the following blog content and generate a list of highly relevant SEO keywords. Focus on terms with high search intent and prioritize short-tail keywords,intermediate-tail keywords.\n\nBlog Content:\n{content}"""
        )
        response = (prompt | llm).invoke({"content": content})
        return [kw.strip() for kw in response.content.strip().split(",") if kw.strip()]
    except Exception as e:
        st.error(f"Error generating SEO keywords: {e}")
        #log_error("Error in suggest_seo_keywords", e)
        return []

# Step 5: Provide SEO Suggestions
def generate_seo_suggestions(content, title, meta_description, keywords):
    """Generates actionable SEO suggestions using the LLM."""
    try:
        prompt = PromptTemplate(
            input_variables=["content", "title", "meta_description", "keywords"],
            template="""Based on the following blog content, title, meta description and SEO keywords, provide actionable suggestions to improve the blog's SEO performance:

          1. **Title** must be (50-55 characters):
             Analyze the Current Title: {title}
             Suggest improvements if necessary.

          2. **Meta Description** must be (150-155 characters):
             Analyze the Current Meta Description: {meta_description}
             Suggest improvements if necessary.

          3. **Content**:
             - Optimize keyword usage and structure (headings, subheadings, internal links).
             - Ensure readability and keyword density.
             - Suggest updates for outdated content.

          Blog Content:
          {content}
        
          SEO Keywords:
          {keywords}
         """
        )
        response = (prompt | llm).invoke({
            "content": content,
            "title": title,
            "meta_description": meta_description,
            "keywords": ", ".join(keywords),
        })
        return response.content.strip()
    except Exception as e:
        st.error(f"Error generating SEO suggestions: {e}")
        #log_error("Error in generate_seo_suggestions", e)
        return "No suggestions available due to an error."

# Step 6: Visualize Keywords as Word Cloud
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
        #log_error("Error in visualize_word_cloud", e)

# Competitor Analysis
def compare_with_competitors(user_keywords, competitor_content):
    """Compares user keywords with competitor blog content."""
    try:
        prompt_template = PromptTemplate(
            input_variables=["user_keywords", "competitor_content"],
            template="""Compare the user's SEO keywords with the competitor's blog content. Identify missing keywords and provide actionable insights.\n\nUser Keywords:\n{user_keywords}\n\nCompetitor Content:\n{competitor_content}"""
        )
        response = (prompt_template | llm).invoke({
            "user_keywords": ", ".join(user_keywords),
            "competitor_content": competitor_content
        })
        return response.content.strip()
    except Exception as e:
        st.error(f"Error in competitor analysis: {e}")
        #log_error("Error in compare_with_competitors", e)
        return "No insights available."

# Streamlit App
def main():
    st.title("Blog SEO Analyzer")

    # Input blog URL
    blog_url = st.text_input("Enter Blog URL:")
    if st.button("Analyze Blog"):
        with st.spinner("Analyzing blog..."):
            content, title, meta_description = retrieve_blog_content(blog_url)
            if not content:
                return

            domain, subdomain = extract_domain_subdomain(blog_url)
            readability = calculate_readability(content)
            keywords = suggest_seo_keywords(content)

            # Display results
            st.subheader("Blog Details")
            st.write(f"**Title:** {title}")
            st.write(f"**Meta Description:** {meta_description}")
            st.write(f"**Domain:** {domain}")
            st.write(f"**Subdomain:** {subdomain}")

            st.subheader("Readability Scores")
            st.metric("Flesch-Kincaid Grade", f"{readability[0]:.2f}" if readability[0] else "N/A")
            st.metric("Flesch Reading Ease", f"{readability[1]:.2f}" if readability[1] else "N/A")

            st.subheader("Suggested Keywords")
            st.write(", ".join(keywords))

            st.subheader("SEO Suggestions")
            seo_suggestions = generate_seo_suggestions(content, title, meta_description, keywords)
            st.text(seo_suggestions)

            st.subheader("Keyword Word Cloud")
            visualize_word_cloud(keywords)

            # Competitor Analysis
            competitor_url = st.text_input("Enter Competitor Blog URL:")
            if st.button("Analyze Competitor Blog"):
                with st.spinner("Analyzing competitor blog..."):
                    competitor_content, _, _ = retrieve_blog_content(competitor_url)
                    if competitor_content:
                        comparison_results = compare_with_competitors(keywords, competitor_content)
                        st.subheader("Competitor Analysis")
                        st.text(comparison_results)

if __name__ == "__main__":
    main()
