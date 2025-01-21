import streamlit as st
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import spacy
import logging
import textstat
from collections import Counter
import pandas as pd

# Constants
API_KEY = "AIzaSyDdMeUzub03ZnrXfpI-c_kJgT1zOQ-lDP4"  # Replace with your actual API key
LLM_MODEL = "gemini-1.5-flash"

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

#Retrieve blog content
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

#Calculate Readability
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

def describe_readability(kincaid_grade, reading_ease):
    if kincaid_grade <= 5:
        grade_description = "Easy to read, suitable for younger audiences."
    elif 6 <= kincaid_grade <= 8:
        grade_description = "Suitable for a general audience."
    elif 9 <= kincaid_grade <= 12:
        grade_description = "High school comprehension."
    elif 13 <= kincaid_grade <= 15:
        grade_description = "College-level comprehension."    
    else:
        grade_description = "Complex content, typically for academic or expert-level readers."
    
    if reading_ease >= 80:
        ease_description = "Very easy to read."
    elif 70 <= reading_ease < 80:
        ease_description = "Easy to read."    
    elif 50 <= reading_ease < 70:
        ease_description = "Fairly easy to read."
    elif 30 <= reading_ease < 50:
        ease_description = "Difficult to read."
    else:
        ease_description = "Very difficult to read, often technical or academic."

    return grade_description, ease_description        

#Extract Keywords
def extract_keywords_from_content(content, top_n=10):
    """Extracts keywords from blog content using NLP."""
    try:
        doc = nlp(content)
        #keywords = [chunk.text.lower() for chunk in doc.noun_chunks]
        #keywords = list(set(keywords))  # Remove duplicates
        keywords = [
            chunk.text.lower()
            for chunk in doc.noun_chunks
            if chunk.text.lower() not in nlp.Defaults.stop_words  # Exclude stopwords
            
        ]
        
        # Count keyword frequencies
        keyword_counts = Counter(keywords)
        
        # Get the top N keywords based on frequency
        top_keywords = [keyword for keyword, _ in keyword_counts.most_common(top_n)]
        return top_keywords
    except Exception as e:
        log_error("Error in extract_top_keywords_from_content", e)
        return []

#Optimize Keywords for SEO
def optimize_seo_keywords(content, page_title, meta_description, url):
    """Optimizes SEO keywords based on structured guidelines."""
    try:
        prompt = PromptTemplate(
            input_variables=["content", "page_title", "meta_description", "url"],
            template="""Analyze the following content based on SEO keyword optimization guidelines. Provide a detailed evaluation of how well the content adheres to each guideline without offering suggestions for improvement:
Each point must be addressed directly, without unnecessary verbosity.
Maintain a clear sequence, following the order of the guidelines.
Focus solely on evaluating adherence to the guidelines. Do not provide suggestions or recommendations for improvement.

Content:
{content}

Page Title:
{page_title}

Meta Description:
{meta_description}

URL:
{url}

SEO Keyword Optimization Guidelines:
    Evaluate whether the content aligns with the target keyword and search intent. Does the keyword fit the content, and is the content optimized for the keyword's search intent?
    Assess if the page meets the user's search intent for the given keyword. Does the content satisfy what the user is looking for?
    Evaluate the inclusion of the primary keyword in the page title. How effectively is it integrated?
    Analyze the effectiveness of the page title in engaging users. Is it likely to attract clicks in search results?
    Assess if the title could benefit from modifiers (e.g., "Best", "Top", "Guide", "2025").
    Evaluate whether the title uses the maximum character length without exceeding it, ensuring it is clear and informative.
    Confirm if the page title is wrapped in an H1 tag and follows correct HTML structure.
    Analyze the inclusion of the primary keyword in the meta description. How well is the keyword used, and is the description compelling for users?
    Evaluate the SEO-friendliness of the URL. Does it include the primary keyword and avoid unnecessary parameters?
    Assess the placement of the primary keyword in the content, particularly in the first sentence.
    Evaluate the keyword density. Is it balanced and consistent with competitor content?
    Analyze the use of variations of the primary keyword and synonyms (LSI keywords). Are these terms used effectively throughout the content?
    Evaluate the readability of the content. Is it structured in a way that is easy for users to read and digest?
    Analyze how user experience factors (e.g., mobile-friendliness, load speed) may impact SEO performance.

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

#Evaluate Content quality of content
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
    Spelling and Grammar: Examine the content for spelling and grammatical errors. Clearly state whether any issues were identified.
    Scannability: Assess the content's readability and formatting. Confirm if headings, bullet points, or other elements make the content easy to scan and consume.
    Readability: Determine if the content is written at an 8th-grade readability level. Highlight any sentences or sections that are overly complex.
    Engagement: Evaluate whether the content effectively captures and maintains the reader's attention throughout. Indicate any sections that might lack engagement.
    Paragraph Structure: Verify that paragraphs are short and structured to avoid dense blocks of text. Mention if any sections deviate from this guideline.
    Heading Structure: Analyze the logical flow of the headings. Confirm whether they guide the reader effectively through the content.
    Heading Clarity: Check if the headings are descriptive and accurately reflect the topic of each section.
    Keyword Usage: Evaluate the use of keyword variations, LSI keywords, or synonyms in the headings and throughout the content. Note the relevance and frequency of their usage.
    Use of Lists: Verify the use of bullet points and numbered lists where applicable. Confirm whether they enhance clarity and structure.
    Originality and Relevance: Validate the originality and relevance of the content. State whether it aligns with current trends and provides up-to-date information.

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

def extract_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    internal_links = []
    external_links = []

    for link in soup.find_all('a', href=True):
        href = link['href']
        # Check if the link is internal or external
        if href.startswith('/') or url in href:
            internal_links.append(href)
        else:
            external_links.append(href)

    return internal_links, external_links

#Evaluate Link quality of content
def evaluate_link_quality(content, internal_links, external_links):
    """Evaluates content quality based on structured guidelines using an LLM."""
    try:
        prompt = PromptTemplate(
            input_variables=["content", "internal_links", "external_links"],
            template="""Analyze the following page content and evaluate it based on the link-related guidelines below. Provide a detailed, structured, and professional evaluation for each point. Ensure the response adheres to the following requirements:

 Each guideline title (e.g., "Internal Links") must be bold in the output for better readability.
 Each point must be addressed directly, without unnecessary verbosity.
 Maintain a clear sequence, following the order of the guidelines.
 Focus solely on evaluating adherence to the guidelines. Do not provide suggestions or recommendations for improvement.

Page Content:
{content}

Link Quality Guidelines:
    Internal Links: Confirm if your page contains internal links. Provide a clear statement about the presence or absence of internal links.List all internal links present in the content.
    Descriptive Anchor Text: Evaluate whether the internal links are using descriptive and relevant anchor text that clearly indicates the target content.
    Internal Link Optimization: Assess if the internal links are optimized based on first link priority (i.e., ensuring that the most important links appear first).
    Breadcrumbs: Verify whether the page includes breadcrumbs to improve navigation and user experience.
    Usefulness of Internal Links: Evaluate if the internal links are genuinely useful to the reader, leading to relevant and valuable content.
    Preferred URLs for Internal Links: Check whether all internal links are using the preferred URLs (i.e., ensuring consistency in linking to canonical versions).
    External Links: Confirm if your page includes external links to relevant sources, partners, or content. List all external links present in the content.
    Affiliate and Sponsored Links: Verify that all affiliate, sponsored, or paid external links use the “NoFollow” tag to comply with SEO best practices.
    External Links Opening in New Window: Evaluate whether all external links are set to open in a new window, ensuring users are not navigated away from the page.
    Broken Links: Confirm if there are any broken links (either internal or external) on the page and specify whether they exist.

The evaluation should deliver a professional, high-quality response that adheres to these standards.        """
        )

        response = (prompt | llm).invoke({
            "content": content,
            "internal_links":", ".join(internal_links) if internal_links else "None found",
            "external_links":", ".join(external_links) if external_links else "None found"
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

            readability_grade,readability_ease = calculate_readability(content)
            grade_description, ease_description = describe_readability(readability_grade, readability_ease)

            st.subheader("Readability Scores")
            st.markdown(f"*Flesch-Kincaid Grade Level*: **{readability_grade}** -***{grade_description}***" if readability_grade else "**Flesch-Kincaid Grade Level**: N/A")
            st.markdown(f"*Flesch Reading Ease*: **{readability_ease}** - ***{ease_description}***" if readability_ease else "**Flesch Reading Ease**: N/A")
 
            #st.markdown("---")

            # Extract keywords from blog content
            extracted_keywords = extract_keywords_from_content(content)
            st.subheader("Extracted Keywords from Blog Content")
            if extracted_keywords:
              #st.write(", ".join(extracted_keywords))
              st.markdown("\n".join([f"{i+1}. {keyword}" for i, keyword in enumerate(extracted_keywords)]))
            else:
              st.warning("No keywords found.")

            # Optimize SEO using extracted keywords
            st.subheader("Keyword Optimization Analysis:")
            seo_suggestions = optimize_seo_keywords(content, title, meta_description, blog_url)
            if seo_suggestions:
              st.markdown("\n".join([f"{i+1}. {suggestion}" for i, suggestion in enumerate(seo_suggestions)]))
              #st.markdown("\n".join([f"- {suggestion}" for suggestion in enumerate(seo_suggestions)]))
            else:
                st.write("No SEO keyword optimization suggestions available.")

            # Evaluate the content quality
            st.subheader("Content Evaluation Analysis:")
            content_quality = evaluate_content_quality(content)
            if content_quality:
                st.markdown("\n".join([f"{i+1}. {evaluation}" for i, evaluation in enumerate(content_quality) if evaluation.strip()]))
            else:
                st.write("No content quality evaluation available.")

            #Evaluate the link quality
        try:
            # Extract links from the blog
            internal_links, external_links = extract_links(blog_url)

            st.subheader("Link Quality Analysis")

            # Consolidate internal links and format as HTML
            unique_internal_links = list(set(internal_links))
            internal_links_formatted = [f'<a href="{link}" target="_blank">{link}</a>' for link in unique_internal_links]

            # Consolidate external links and format as HTML
            unique_external_links = list(set(external_links))
            external_links_formatted = [f'<a href="{link}" target="_blank">{link}</a>' for link in unique_external_links]

            # Create a DataFrame for the table
            data = {
                "Category": ["Internal Links", "External Links"],
                "Links": [", ".join(internal_links_formatted), ", ".join(external_links_formatted)],
                "Count": [len(unique_internal_links), len(unique_external_links)],
            }
            df = pd.DataFrame(data)

            # Display table
            st.markdown("### Link Analysis Table")
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

            # Evaluate link quality
            link_content = " ".join(internal_links + external_links)
            evaluation = evaluate_link_quality(link_content, unique_internal_links, unique_external_links)

            # Display evaluation
            st.markdown("### Link Quality Evaluation")
            st.markdown (evaluation)
        except Exception as e:
            st.error(f"Error analyzing the blog: {e}") 

if __name__ == "__main__":
    main()
