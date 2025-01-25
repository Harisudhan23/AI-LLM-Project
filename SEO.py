import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import spacy
import logging
import textstat
from collections import Counter
import pandas as pd
import re  # Added for regex operations
from urllib.parse import urlparse, urljoin

# Constants
API_KEY = "AIzaSyBzSFL43Im7fIv-UGD9WTV4RitWG4VQC0g"  # Replace with your actual API key
LLM_MODEL = "gemini-2.0-flash-exp"

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

# Function to remove zero-width characters
def remove_zw_chars(text):
    """Removes zero-width joiner and other related characters."""
    return re.sub(r"[\u200B-\u200F\u2060-\u206F\uFEFF]", "", text)

def clean_text(text):
    """Removes common encoding artifacts and extra spaces from text."""
    text = text.replace("Â ", " ")  # Replace "Â " with a normal space
    text = text.replace("â€™", "'")  # Replace "â€™" with an apostrophe
    text = text.replace("Â", "")    #Removes Â characters
    text = text.replace("  ", " ") #Remove Double Spaces
    return text


def print_text_before_llm(text, label="Text Before LLM:"):
    """Prints the given text with a label. Useful for debugging.

    Args:
        text (str): The text to print.
        label (str, optional): A label to identify the printed text. Defaults to "Text Before LLM:".
    """
    print("=" * 40)
    print(f"{label}")
    print("-" * 40)
    print(text)
    print("=" * 40)


#Retrieve blog content
def retrieve_blog_content(url):
    """Fetches and parses blog content from a given URL, including headings and list items."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response.encoding = "utf-8" #explicitly set encoding
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from <p>, <h1>, <h2>, <h3>, and <li> tags
        elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
        text_content = " ".join([elem.text for elem in elements])

        content = clean_text(text_content) #Clean retrieved text

        title = soup.title.string.strip() if soup.title else "No title found"
        meta_tag = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
        meta_description = meta_tag["content"].strip() if meta_tag else "No meta description found"

        if not content.strip():
            raise ValueError("Blog content is empty or could not be retrieved.")

        return content, clean_text(title), clean_text(meta_description)
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
    elif 6 <= kincaid_grade <= 9:
        grade_description = "Suitable for a general audience."
    elif 9 <= kincaid_grade <= 12:
        grade_description = "High school comprehension."
    elif 12 <= kincaid_grade <= 15:
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
def extract_keywords_from_content(content, top_n=20):
    """Extracts keywords from blog content using NLP."""
    try:
        # Print the text before sending to LLM
        cleaned_content = clean_text(content)
        print_text_before_llm(cleaned_content, "Text Before Keyword Extraction LLM:")
        doc = nlp(cleaned_content)
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
def optimize_seo_keywords(content, page_title, meta_description, url, llm):
    """Optimizes SEO keywords based on structured guidelines."""
    try:
        prompt = PromptTemplate(
            input_variables=["content", "page_title", "meta_description", "url"],
            template="""Analyze the following content based on SEO keyword optimization guidelines. Provide a detailed evaluation of how well the content adheres to each guideline.Focus solely on factual analysis. Do not include any recommendations or suggestions.

    Format the analysis output as follows:

    Keyword and Search Intent Alignment: [Your analysis here]

    Primary Keyword in Page Title: [Your analysis here]

    Page Title Engagement: [Your analysis here]

    Page Title Modifiers: [Your analysis here]

    Page Title Character Length: [Your analysis here]

    Page Title HTML Structure: [Your analysis here. If you can't assess, state that]

    Primary Keyword in Meta Description: [Your analysis here]

    Primary Keyword in URL: [Your analysis here]

    Primary Keyword in First Sentence: [Your analysis here]

    Keyword Density: [Your analysis here]

    Top 5 Keywords Distribution: [Your analysis here]

    Variations and LSI Keywords: [Your analysis here]

    Readability: [Your analysis here]

    User Experience Factors: [Your analysis here. If you can't assess, state that]

    Content:
    {content}

    Page Title:
    {page_title}

    Meta Description:
    {meta_description}

    URL:
    {url}

    SEO Keyword Optimization Guidelines:
        Evaluate the content's alignment with the target keyword and search intent, ensuring it fits the content and satisfies user expectations.
        Assess the integration of the primary keyword in the page title and how well it is optimized for search intent.
        Analyze the effectiveness of the page title in engaging users and its likelihood of attracting clicks in search results.
        Determine if the page title could benefit from modifiers like "Best," "Top," "Guide," or "2025."
        Confirm whether the page title utilizes the maximum character length without exceeding it while remaining clear and informative.
        Verify that the page title is wrapped in an H1 tag and follows correct HTML structure.
        Evaluate the inclusion and usage of the primary keyword in the meta description and its effectiveness in compelling users.
        Determine if the primary keyword is present in the URL, and assess whether the URL structure is lean and optimized for SEO.
        Analyze the placement of the primary keyword in the content, especially in the first sentence.
        Assess the keyword density, ensuring it is balanced and consistent with competitor content.
        Ensure that top 5 keywords are distributed across various article sections.
        Evaluate the use of variations of the primary keyword and synonyms (LSI keywords) throughout the content for effective optimization.
        Assess the readability of the content, ensuring it is structured for easy reading and digestion by users.
        Analyze user experience factors, such as mobile-friendliness and load speed, that could impact SEO performance

    The evaluation should deliver a professional, high-quality response that adheres to these standards.
    """
        )
        
        cleaned_content = clean_text(content)
        cleaned_page_title = clean_text(page_title)
        cleaned_meta_description = clean_text(meta_description)
        cleaned_url = clean_text(url)

        # Print the text before sending to LLM
        print_text_before_llm(f"Content: {cleaned_content}, Page Title: {cleaned_page_title}, Meta Description: {cleaned_meta_description}, URL: {cleaned_url}", "Text Before SEO Optimization LLM:")

        response = llm.invoke(prompt.format(
            content=cleaned_content,
            page_title=cleaned_page_title,
            meta_description=cleaned_meta_description,
            url=cleaned_url,
        ))
        # Process and return the detailed evaluation as a list
        return [remove_zw_chars(line) for line in response.content.strip().split("\n")]
    except Exception as e:
        st.error(f"Error optimizing SEO keywords: {e}")
        log_error("Error in optimize_seo_keywords", e)
        return []

#Evaluate Content quality of content
def evaluate_content_quality(content, llm):
    """Evaluates content quality based on structured guidelines using an LLM."""
    try:
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""Analyze the following content based on content quality guidelines. Provide a detailed evaluation of how well the content adheres to each guideline. Focus solely on factual analysis. Do not include any recommendations or suggestions.

            Each guideline title (e.g., "Spelling and Grammar") must be bold in the output for better readability.
            Each point must be addressed directly, without unnecessary verbosity.
            Maintain a clear sequence, following the order of the guidelines.
            Focus solely on evaluating adherence to the guidelines. Provide a clear and actionable analysis for each point.
            When providing your analysis of the "Spelling and Grammar" guideline, make note if there is awkward phrasing, extra spaces between words or periods, or other spacing errors.

            Blog Content:
            {content}

            Content Quality Guidelines:
                Spelling and Grammar: Examine the content to check for spacing, spelling and grammatical errors using tools like Grammarly. Clearly state whether any issues were identified.
                Scannability: Assess the content's readability and formatting. Confirm if headings, bullet points, or other elements make the content easy to scan and consume.
                Readability: Ensure the content is written at an 8th-grade readability level. Highlight any sentences or sections that are overly complex.
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

        # Clean the text *before* printing and sending to LLM
        cleaned_content = clean_text(content)
        print_text_before_llm(cleaned_content, "Text Before Content Quality LLM:")

        response = llm.invoke(prompt.format(content = cleaned_content))
       # Process and return the detailed evaluation as a list
        return [remove_zw_chars(line) for line in response.content.strip().split("\n")]
    except Exception as e:
        st.error(f"Error evaluating content quality: {e}")
        log_error("Error in evaluate_content_quality", e)
        return []

prompt_template = PromptTemplate.from_template("""
Analyze the following page content and evaluate its link structure according to the guidelines provided. Provide your response in the strict format specified below.

**Internal Links**
*   [Analysis of internal links - is it present or not]

**Descriptive Anchor Text**
*   [Analysis of descriptive anchor text for internal links]

**Internal Link Optimization**
*  [Analysis of internal link prioritization, are important links first]

**Breadcrumbs**
*   [Analysis of the presence or absence of breadcrumbs]

**Usefulness of Internal Links**
*  [Analysis of the usefulness of internal links]

**Preferred URLs for Internal Links**
*   [Analyze the page content. Are all internal links using the preferred URLs (i.e., ensuring consistency in linking to canonical versions) Answer, stating if it can be determined from provided content or not and your analysis]

**External Links**
*   [Analyze the page content. Determine if the page *likely* includes external links to relevant sources, partners, or content. Base your answer on what is likely given the context. Provide an analysis, including if it is highly likely or not and why.]

**Affiliate and Sponsored Links**
*   [Analyze the page content. Determine if all affiliate, sponsored, or paid external links are *likely* using the “NoFollow” tag to comply with SEO best practices. Based on the context, determine how likely it is that these exist and infer if they are likely to be 'no follow'. State if this is based on an inference from the provided content or if it is something that can not be determined.]

**External Links Opening in New Window**
*   [Analyze the page content. Determine if, based on best practices, the external links are *likely* set to open in a new window. State if this is based on an inference or if it cannot be determined from content provided.]

**Broken Links**
*   [Analyze the page content. Based on the page content, infer if there might be any broken links (either internal or external) on the page. Provide analysis and explain, and if this is not something that is possible to ascertain from the content, say that.]

Page Content:
{content}
""")

def scrape_page_content(url):
    """Scrapes the HTML content of a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises an exception for bad status codes
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, "html.parser")
        text_content = soup.get_text(separator=" ", strip=True)
        return clean_text(text_content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error accessing URL: {e}")
        log_error("Error in scrape_page_content", e)
        return None


def analyze_url(content, llm):
    """Sends a prompt to the LLM and returns the analysis."""
    prompt = prompt_template.format(content=clean_text(content))
    response = llm.invoke(prompt)
    return [remove_zw_chars(line) for line in response.content.strip().split("\n")]


# Streamlit App
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
            page_content = scrape_page_content(blog_url)
            if not page_content:
                st.error("Failed to retrieve page content")
                return

            # Retrieve blog content
            content, title, meta_description = retrieve_blog_content(blog_url)
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

            #Evaluate the link quality
            with st.expander("Link Evaluation"):
                try:
                    if page_content:
                       analysis = analyze_url(page_content, llm)
                       st.markdown("\n".join([f" {evaluation}" for evaluation in analysis]))


                except Exception as e:
                    st.error(f"Error analyzing the blog: {e}")

if __name__ == "__main__":
    main()