import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
from langchain.prompts import PromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import re
import logging
import os
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

API_KEY = "AIzaSyBzSFL43Im7fIv-UGD9WTV4RitWG4VQC0g"  # Replace with your actual API key
LLM_MODEL = "gemini-2.0-flash-exp"

# Initialize LLM
llm = ChatGoogleGenerativeAI(api_key=API_KEY, model=LLM_MODEL)
# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

try:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)

except ValueError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.error(f"Invalid log level provided using default value: 'INFO'")



def clean_text(text):
    """Cleans up text by removing extra spaces, newlines and HTML tags."""
    if not text:
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def scrape_page_content(url):
    """
    Scrapes the content of a webpage, returning a BeautifulSoup object.
    """
    logger.info(f"Starting scraping process for URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        logger.info(f"Scraping process completed successfully for URL: {url}")
        return soup
    except requests.exceptions.RequestException as e:
         logger.error(f"Error fetching the URL: {e}")
         st.error(f"Error fetching the URL: {e}")
         return None


def retrieve_blog_content(url, soup):
    """
    Extracts blog content, title, and meta description from BeautifulSoup object.
    """
    logger.info(f"Starting retrieval of blog content for URL: {url}")
    try:
        data = {
            "title": "",
            "content": "",
            "meta_description": ""
        }
        
        title_element = soup.find('h1')
        if title_element:
           data["title"] = title_element.text.strip()
        
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
           data["meta_description"] = meta_desc.get('content')
           
        cleaned_content = ""
        content_elements = soup.find_all(['h2', 'h3', 'p', 'ul','img','a', 'div'])
        for element in content_elements:
            if element.name == 'p':
                cleaned_content +=  " " + clean_text(element.get_text(strip=True))
            elif element.name == 'h2' or element.name == 'h3':
                cleaned_content += " " + clean_text(element.get_text(strip=True))
            elif element.name == 'ul':
                 list_items = [li.get_text(strip=True) for li in element.find_all('li')]
                 cleaned_content += " " + " ".join(list_items)
            elif element.name == 'div' and "feature-check" in element.get('class',[]):
                check_elements = element.find_all('p')
                if check_elements:
                    cleaned_content += " " + " ".join([ check.get_text(strip=True) for check in check_elements ])
        
        data["content"] = cleaned_content
        logger.info(f"Retrieval of blog content completed successfully for URL: {url}")
        return data["content"], data["title"], data["meta_description"]

    except Exception as e:
        logger.error(f"Error in retrieving blog content for URL: {url} error is {e}")
        st.error(f"An error occurred: {e}")
        return None, None, None


def calculate_readability(content):
    """
     Calculates the Flesch-Kincaid Grade Level and Flesch Reading Ease scores.
    """
    logger.info(f"Calculating readability scores")
    try:
        if not content:
            logger.warning(f"No content available to calculate readability")
            return None, None
    
        sentences = nltk.sent_tokenize(content)
        words = nltk.word_tokenize(content)
    
        if not sentences or not words:
            logger.warning(f"Could not process sentences and words of content")
            return None, None
        
        num_sentences = len(sentences)
        num_words = len(words)

        syllables = 0
        for word in words:
           for char in word:
                if char.lower() in "aeiou":
                    syllables += 1
            
        if num_sentences == 0 or num_words == 0:
            logger.warning("Number of sentences or words is 0")
            return None, None
        
        flesch_reading_ease = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllables / num_words)
        flesch_kincaid_grade = 0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59

        logger.info(f"Readability scores calculation completed successfully")
        return round(flesch_kincaid_grade, 2) , round(flesch_reading_ease, 2)
    
    except Exception as e:
        logger.error(f"An error occurred while calculating readability: {e}")
        return None, None
    
    
def describe_readability(readability_grade, readability_ease):
    """Provides descriptions of readability scores."""
    logger.info(f"Providing description of the readability scores")
    if readability_grade is None:
        grade_description = "N/A"
    elif readability_grade < 7:
      grade_description = "Very easy to read"
    elif readability_grade < 10:
        grade_description = "Easy to read"
    elif readability_grade < 13:
        grade_description = "Normal to read"
    elif readability_grade < 16:
        grade_description = "Difficult to read"
    else:
        grade_description = "Very difficult to read"

    if readability_ease is None:
        ease_description = "N/A"
    elif readability_ease > 80:
        ease_description = "Very Easy"
    elif readability_ease > 70:
        ease_description = "Easy"
    elif readability_ease > 60:
        ease_description = "Standard"
    elif readability_ease > 50:
        ease_description = "Fairly Difficult"
    else:
      ease_description = "Difficult"
    
    logger.info(f"Descriptions provided successfully")
    return grade_description, ease_description
    

def extract_keywords_from_content(content, top_n=10):
    """Extracts top keywords from content using NLTK."""
    logger.info(f"Extracting keywords from content")
    try:
        if not content:
            logger.warning(f"No content available to extract keywords from")
            return None
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(content)
        words = [word.lower() for word in words if word.isalpha()]
        filtered_words = [word for word in words if word not in stop_words]

        word_counts = Counter(filtered_words)
        most_common_words = [word for word, count in word_counts.most_common(top_n)]
        logger.info(f"Keywords extracted successfully from content")
        return most_common_words
    
    except Exception as e:
        logger.error(f"Error during keyword extraction from content: {e}")
        return None
    

def optimize_seo_keywords(content, page_title, meta_description, url, llm):
    """Optimizes SEO keywords based on structured guidelines."""
    logger.info(f"Starting SEO keyword optimization process.")
    try:
        prompt = PromptTemplate(
            input_variables=["content", "page_title", "meta_description", "url"],
            template="""
            You are an expert SEO analyst. You will be provided with the content of a blog post, its page title, meta description, and URL. Your task is to analyze this data based on the SEO keyword optimization guidelines provided below. Your analysis should be factual and direct, stating whether or not each guideline is met, and provide a concise explanation for your assessment, drawing directly from the provided content. Avoid including any recommendations or suggestions beyond your analysis. If information to evaluate a guideline is unavailable, please respond with "Insufficient Information". 
        
            Format the analysis output as follows, with each section starting on a new line:

            **Keyword and Search Intent Alignment:** [Your analysis here]

            **Primary Keyword in Page Title:** [Your analysis here]

            **Page Title Engagement:** [Your analysis here]

            **Page Title Modifiers:** [Your analysis here]

            **Page Title Character Length:** [Your analysis here]

            **Page Title HTML Structure:** [Your analysis here]

            **Primary Keyword in Meta Description:** [Your analysis here]

            **Primary Keyword in URL:** [Your analysis here]

            **Primary Keyword in First Sentence:** [Your analysis here]

            **Keyword Density:** [Your analysis here]

            **Top 5 Keywords Distribution:** [Your analysis here]

            **Variations and LSI Keywords:** [Your analysis here]

            **Content:**
            {content}

            **Page Title:**
            {page_title}

            **Meta Description:**
            {meta_description}

            **URL:**
            {url}

            **SEO Keyword Optimization Guidelines:**
                *   **Keyword and Search Intent Alignment:** Evaluate the content's alignment with the target keyword and search intent, ensuring it fits the content and satisfies user expectations.
                *   **Primary Keyword in Page Title:** Assess the integration of the primary keyword in the page title and how well it is optimized for search intent.
                *   **Page Title Engagement:** Analyze the effectiveness of the page title in engaging users and its likelihood of attracting clicks in search results.
                *   **Page Title Modifiers:** Assess whether the page title could benefit from the inclusion of a temporal modifier, such as a year (e.g., '2025'), considering its current click-worthiness and character length.
                *   **Page Title Character Length:** Confirm whether the page title utilizes the maximum character length without exceeding it while remaining clear and informative.
                *   **Page Title HTML Structure:** Verify that the page title is wrapped in an H1 tag and follows correct HTML structure.
                *   **Primary Keyword in Meta Description:** Evaluate the inclusion and usage of the primary keyword in the meta description and its effectiveness in compelling users.
                *   **Primary Keyword in URL:** Determine if the primary keyword is present in the URL, and assess whether the URL structure is lean and optimized for SEO.
                *   **Primary Keyword in First Sentence:** Analyze the placement of the primary keyword in the content, especially in the first sentence.
                *   **Keyword Density:** Assess the keyword density, ensuring it is balanced and consistent with competitor content.
                *   **Top 5 Keywords Distribution:** Ensure that top 5 keywords are distributed across various article sections.
                *   **Variations and LSI Keywords:** Evaluate the use of variations of the primary keyword and synonyms (LSI keywords) throughout the content for effective optimization.
            
            The evaluation should deliver a professional, high-quality response that adheres to these standards.
    """
        )

        response = llm.invoke(prompt.format(content=content, page_title=page_title, meta_description=meta_description, url=url))
        logger.info(f"SEO keyword optimization process completed successfully.")
        return response.split("\n")
    except Exception as e:
        logger.error(f"Error during SEO keyword optimization: {e}")
        return [f"An error occurred during SEO keyword optimization: {e}"]
        
def evaluate_content_quality(content, llm):
    """Evaluates content quality based on structured guidelines using an LLM."""
    logger.info(f"Starting content quality evaluation process.")
    try:
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""
             You are a content quality expert. You will be provided with the content of a blog post. Your task is to analyze the blog post content based on content quality guidelines. Your analysis should be factual and direct, stating whether or not each guideline is met, and provide a concise explanation for your assessment, drawing directly from the provided content. Avoid including any recommendations or suggestions beyond your analysis. If information to evaluate a guideline is unavailable, please respond with "Insufficient Information". 

            Each guideline title (e.g., "**Spelling and Grammar**") must be bold in the output for better readability.
            Each point must be addressed directly, without unnecessary verbosity.
            Maintain a clear sequence, following the order of the guidelines.
            Focus solely on evaluating adherence to the guidelines. Provide a clear and actionable analysis for each point.
            When providing your analysis of the **Spelling and Grammar** guideline, make note if there is awkward phrasing, extra spaces between words or periods, or other spacing errors.

            **Blog Content:**
            {content}

            **Content Quality Guidelines:**
                *   **Spelling and Grammar**: Examine the content to check for spacing, spelling and grammatical errors using tools like Grammarly. Clearly state whether any issues were identified.
                *   **Scannability**: Assess the content's readability and formatting. Confirm if headings, bullet points, or other elements make the content easy to scan and consume.
                *   **Readability**: Ensure the content is written at an 8th-grade readability level. Highlight any sentences or sections that are overly complex.
                *  **Engagement**: Evaluate whether the content effectively captures and maintains the reader's attention throughout. Indicate any sections that might lack engagement.
                *   **Paragraph Structure**: Verify that paragraphs are short and structured to avoid dense blocks of text. Mention if any sections deviate from this guideline.
                *   **Heading Structure**: Analyze the logical flow of the headings. Confirm whether they guide the reader effectively through the content.
                *   **Heading Clarity**: Check if the headings are descriptive and accurately reflect the topic of each section.
                *   **Keyword Usage**: Evaluate the use of keyword variations, LSI keywords, or synonyms in the headings and throughout the content. Note the relevance and frequency of their usage.
                *   **Use of Lists**: Verify the use of bullet points and numbered lists where applicable. Confirm whether they enhance clarity and structure.
                *   **Originality and Relevance**: Validate the originality and relevance of the content. State whether it aligns with current trends and provides up-to-date information.

            The evaluation should deliver a professional, high-quality response that adheres to these standards.
        """
        )

        response = llm.invoke(prompt.format(content=content))
        logger.info(f"Content quality evaluation process completed successfully.")
        return response.split("\n")
    except Exception as e:
        logger.error(f"Error during content quality evaluation: {e}")
        return [f"An error occurred during content quality evaluation: {e}"]

def main():
    st.title("Blog SEO Analyzer")
    st.markdown("---")

    # CSS to style subheadings
    st.markdown("""
        <style>
        .st-expander h4 {
            font-size: 1.5em;
            font-weight: bold;
            color: #333; /* You can choose a color */
            padding-bottom: 0.5em;
        }
        </style>
        """, unsafe_allow_html=True)

    # Input blog URL
    blog_url = st.text_input("Enter Blog URL:")
    if st.button("Analyze"):
        with st.spinner("Analyzing blog..."):
            soup = scrape_page_content(blog_url)
            if not soup:
                st.error("Failed to retrieve page content")
                return

            # Retrieve blog content, title, meta description
            content, title, meta_description = retrieve_blog_content(blog_url, soup)
            if not content:
                return

            readability_grade, readability_ease = calculate_readability(content)
            grade_description, ease_description = describe_readability(readability_grade, readability_ease)

            with st.expander("Readability Scores"):
                if readability_grade:
                    st.markdown(
                        f"""
                        <div style="font-size:22px; font-weight:bold;">Flesch-Kincaid Grade Level</div>
                        <div style="font-size:16px; color:gray;">{readability_grade} - {grade_description}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("**Flesch-Kincaid Grade Level**: N/A")

                if readability_ease:
                    st.markdown(
                        f"""
                        <div style="font-size:22px; font-weight:bold;">Flesch Reading Ease</div>
                        <div style="font-size:16px; color:gray;">{readability_ease} - {ease_description}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                     st.markdown("**Flesch Reading Ease**: N/A")

            # Extract keywords from blog content
            extracted_keywords = extract_keywords_from_content(content)
            with st.expander("Extracted Keywords from Blog Content"):
                if extracted_keywords:
                    st.markdown("\n".join([f"{i+1}. {keyword}" for i, keyword in enumerate(extracted_keywords)]))
                else:
                   st.warning("No keywords found.")

            # Optimize SEO using extracted keywords
            
            with st.expander("Keyword Optimization Analysis"):
                seo_suggestions = optimize_seo_keywords(content, title, meta_description, blog_url, llm)
                if seo_suggestions:
                    st.markdown("\n".join([f" {suggestion}" for suggestion in seo_suggestions]))
                else:
                    st.write("No SEO keyword optimization suggestions available.")

            # Evaluate the content quality
            with st.expander("Content Evaluation Analysis"):
                content_quality = evaluate_content_quality(content, llm)
                if content_quality:
                    st.markdown("\n".join([f" {evaluation}" for evaluation in content_quality]))
                else:
                   st.write("No content quality evaluation available.")

if __name__ == "__main__":
    main()