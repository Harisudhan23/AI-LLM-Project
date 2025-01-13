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
            template="""Analyze the following content based on SEO keyword optimization guidelines. Provide a detailed evaluation of how well the content adheres to each guideline without offering suggestions for improvement:
        
            Content:
            {content}

            Page Title:
            {page_title}

            Meta Description:
            {meta_description}

            URL:
            {url}

            SEO Keyword Optimization Guidelines:
            1. Evaluate whether the content aligns with the target keyword and search intent. Does the keyword fit the content, and is the content optimized for the keyword's search intent?
            2. Assess if the page meets the user's search intent for the given keyword. Does the content satisfy what the user is looking for?
            3. Evaluate the inclusion of the primary keyword in the page title. How effectively is it integrated?
            4. Analyze the effectiveness of the page title in engaging users. Is it likely to attract clicks in search results?
            5. Assess if the title could benefit from modifiers (e.g., "Best", "Top", "Guide", "2025").
            6. Evaluate whether the title uses the maximum character length without exceeding it, ensuring it is clear and informative.
            7. Confirm if the page title is wrapped in an H1 tag and follows correct HTML structure.
            8. Analyze the inclusion of the primary keyword in the meta description. How well is the keyword used, and is the description compelling for users?
            9. Evaluate the SEO-friendliness of the URL. Does it include the primary keyword and avoid unnecessary parameters?
            10. Assess the placement of the primary keyword in the content, particularly in the first sentence.
            11. Evaluate the keyword density. Is it balanced and consistent with competitor content?
            12. Analyze the use of variations of the primary keyword and synonyms (LSI keywords). Are these terms used effectively throughout the content?
            13. Evaluate the readability of the content. Is it structured in a way that is easy for users to read and digest?
            14. Analyze how user experience factors (e.g., mobile-friendliness, load speed) may impact SEO performance.

            Based on this analysis, provide a thorough evaluation of how well the content adheres to the above SEO keyword optimization guidelines.
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
            template="""Analyze the following blog content and evaluate it based on the content guidelines below. Provide a detailed, structured, and professional evaluation for each point. Ensure the response adheres to the following requirements:

 Each guideline title (e.g., "Spelling and Grammar") must be bold in the output for better readability.
 Each point must be addressed directly, without unnecessary verbosity.
 Maintain a clear sequence, following the order of the guidelines.
 Focus solely on evaluating adherence to the guidelines. Do not provide suggestions or recommendations for improvement.

Blog Content:
{content}

Content Quality Guidelines:
1. Spelling and Grammar: Examine the content for spelling and grammatical errors. Clearly state whether any issues were identified.
2. Scannability: Assess the content's readability and formatting. Confirm if headings, bullet points, or other elements make the content easy to scan and consume.
3. Readability: Determine if the content is written at an 8th-grade readability level. Highlight any sentences or sections that are overly complex.
4. Engagement: Evaluate whether the content effectively captures and maintains the reader's attention throughout. Indicate any sections that might lack engagement.
5. Paragraph Structure: Verify that paragraphs are short and structured to avoid dense blocks of text. Mention if any sections deviate from this guideline.
6. Heading Structure: Analyze the logical flow of the headings. Confirm whether they guide the reader effectively through the content.
7. Heading Clarity: Check if the headings are descriptive and accurately reflect the topic of each section.
8. Keyword Usage: Evaluate the use of keyword variations, LSI keywords, or synonyms in the headings and throughout the content. Note the relevance and frequency of their usage.
9. Use of Lists: Verify the use of bullet points and numbered lists where applicable. Confirm whether they enhance clarity and structure.
10. Originality and Relevance: Validate the originality and relevance of the content. State whether it aligns with current trends and provides up-to-date information.

The evaluation should deliver a professional, high-quality response that adheres to these standards.
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

def evaluate_link_quality(content):
    """Evaluates content quality based on structured guidelines using an LLM."""
    try:
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""Analyze the following page content and evaluate it based on the link-related guidelines below. Provide a detailed, structured, and professional evaluation for each point. Ensure the response adheres to the following requirements:

 Each guideline title (e.g., "Internal Links") must be bold in the output for better readability.
 Each point must be addressed directly, without unnecessary verbosity.
 Maintain a clear sequence, following the order of the guidelines.
 Focus solely on evaluating adherence to the guidelines. Do not provide suggestions or recommendations for improvement.

Page Content:
{content}

Link Quality Guidelines:
1. Internal Links: Confirm if your page contains internal links. Provide a clear statement about the presence or absence of internal links.
2. Descriptive Anchor Text: Evaluate whether the internal links are using descriptive and relevant anchor text that clearly indicates the target content.
3. Internal Link Optimization: Assess if the internal links are optimized based on first link priority (i.e., ensuring that the most important links appear first).
4. Breadcrumbs: Verify whether the page includes breadcrumbs to improve navigation and user experience.
5. Usefulness of Internal Links: Evaluate if the internal links are genuinely useful to the reader, leading to relevant and valuable content.
6. Preferred URLs for Internal Links: Check whether all internal links are using the preferred URLs (i.e., ensuring consistency in linking to canonical versions).
7. External Links: Confirm if your page includes external links to relevant sources, partners, or content.
8. Affiliate and Sponsored Links: Verify that all affiliate, sponsored, or paid external links use the “NoFollow” tag to comply with SEO best practices.
9. External Links Opening in New Window: Evaluate whether all external links are set to open in a new window, ensuring users are not navigated away from the page.
10. Broken Links: Confirm if there are any broken links (either internal or external) on the page and specify whether they exist.

The evaluation should deliver a professional, high-quality response that adheres to these standards.
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
    st.markdown("---")

    # Input blog URL
    blog_url = st.text_input("Enter Blog URL:")
    if st.button("Analyze"):
        with st.spinner("Analyzing blog..."):
            # Retrieve blog content
            content, title, meta_description = retrieve_blog_content(blog_url)
            if not content:
                return

            readability = calculate_readability(content)
            st.subheader("Readability Scores")
            st.metric("Flesch-Kincaid Grade", f"{readability[0]:.2f}" if readability[0] else "N/A")
            st.metric("Flesch Reading Ease", f"{readability[1]:.2f}" if readability[1] else "N/A")  
            st.markdown("---")

            # Extract keywords from blog content
            extracted_keywords = extract_keywords_from_content(content)
            st.subheader("Extracted Keywords from Blog Content")
            #st.write(", ".join(extracted_keywords))
            st.markdown("- " + "\n- ".join(extracted_keywords))

            # Optimize SEO using extracted keywords
            st.subheader("Keyword Optimization Analysis:")
            seo_suggestions = optimize_seo_keywords(content, title, meta_description, blog_url)
            for seo in seo_suggestions:
                st.write(f"- {seo}")

            # Evaluate the content quality
            st.subheader("Content Evaluation Analysis:")
            content_suggestion = evaluate_content_quality(content)
            for content_suggest in content_suggestion:
                st.write(f"- {content_suggest}")

            st.subheader("Link Quality Analysis:")
            link_suggestion = evaluate_link_quality(content)
            for link in link_suggestion:
                st.write(f"- {link}")    

if __name__ == "__main__":
    main()
