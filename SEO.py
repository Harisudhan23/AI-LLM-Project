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
import html

# Constants
API_KEY = "AIzaSyCTwrc_zTvlfkwXKs-QJOK61tGZfEpUbzQ"  # Replace with your actual API key
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

def clean_placeholder_text(content, url=None):
    """Cleans unwanted placeholder text and special characters from content, preserving contractions."""
    unwanted_patterns = [
        r'Lorem ipsum.*?(\.|。)',       # Placeholder text
        r'Sample content.*?(\.|。)',    # Sample content
        r'[\u2000-\u206F\uFEFF]',      # Unwanted Unicode whitespace characters
        r'[^\w\s.,!?\'\-\–/]',          # Non-standard special characters (excluding apostrophe, hyphen and en dash)
        r'\[.*?\]',                    # Text inside square brackets
        r'\(.*?\)'                     # Text inside parentheses
    ]

    url_specific_patterns = {
        'example.com': [r'Example text.*?(\.|。)', r'Extra placeholder.*?(\.|。)'],
        'another.com': [r'Special content.*?(\.|。)']
    }

    # Apply general patterns
    for pattern in unwanted_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)

    # Apply URL-specific patterns
    if url and url in url_specific_patterns:
        for pattern in url_specific_patterns[url]:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # Replace en dash and em dash with hyphen
    content = content.replace("–", "-")
    content = content.replace("—", "-")

    # Normalize multiple spaces to single spaces (and remove surrounding whitespace)
    content = re.sub(r'\s+', ' ', content).strip()

    return content.strip()

def scrape_page_content(url):
    """Scrapes the HTML content of a given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises an exception for bad status codes
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.content, "html.parser")
        return soup
    except requests.exceptions.RequestException as e:
        st.error(f"Error accessing URL: {e}")
        log_error("Error in scrape_page_content", e)
        return None

#Retrieve blog content
def retrieve_blog_content(url, soup):
    """Fetches and parses blog content from a given URL, including headings and list items, while filtering out footers and sidebars."""
    try:
        # Example selectors, adjust these based on the specific site
        main_content_selector = 'article'  #Common selector for main article
        #Remove sidebars using the "aside" or "sidebar" selectors
        sidebar_selector = 'aside, div[id*="sidebar"]' #Combined selector
        #Remove footers using the "footer" selector
        footer_selector = 'footer' #Footer selector

       #Find and remove the sidebars and footers
        for element in soup.select(sidebar_selector):
           element.decompose()
        for element in soup.select(footer_selector):
           element.decompose()

        # Extract text from <p>, <h1>, <h2>, <h3>, and <li> tags
        elements = soup.find_all(['p', 'h1', 'h2', 'h3','h4','h5','h6', 'li'])
        text_content = " ".join([elem.text for elem in elements])
        content = clean_placeholder_text(text_content, url) #Clean retrieved text
        #content = remove_placeholder_unicode(content)
        title = soup.title.string.strip() if soup.title else "No title found"
        meta_tag = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
        meta_description = meta_tag["content"].strip() if meta_tag else "No meta description found"

        if not content.strip():
            raise ValueError("Blog content is empty or could not be retrieved.")

        return content, clean_placeholder_text(title), clean_placeholder_text(meta_description)

    except requests.exceptions.RequestException as e:
      st.error(f"Error retrieving blog content: {e}")
      log_error("RequestException in retrieve_blog_content", e)
      return None, None, None
    except ValueError as e:
      st.error(f"Content Error: {e}")
      log_error("ValueError in retrieve_blog_content", e)
      return None, None, None
    except Exception as e:
      st.error(f"Error in retrieve_blog_content: {e}")
      log_error("Error in retrieve_blog_content", e)
      return None, None, None

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
    """Describes readability and provides suggestions."""
    grade_description = ""
    ease_description = ""
    grade_suggestion = ""
    ease_suggestion = ""


    if kincaid_grade <= 5:
        grade_description = "Easy to read, suitable for younger audiences."
    elif 6 <= kincaid_grade <= 9:
        grade_description = "Suitable for a general audience."
    elif 9 <= kincaid_grade <= 12:
        grade_description = "High school comprehension."
        grade_suggestion = "Consider simplifying some sentences or using less technical jargon if the target audience is not academic."
    elif 12 <= kincaid_grade <= 15:
        grade_description = "College-level comprehension."
        grade_suggestion = "Reduce sentence complexity and use simpler vocabulary for a broader readership."
    else:
        grade_description = "Complex content, typically for academic or expert-level readers."
        grade_suggestion = "Simplify sentences, use common words, and clarify complex topics for a wider audience."


    if reading_ease >= 80:
        ease_description = "Very easy to read."
    elif 70 <= reading_ease < 80:
        ease_description = "Easy to read."
    elif 50 <= reading_ease < 70:
        ease_description = "Fairly easy to read."
        ease_suggestion = "Consider using shorter sentences and simpler vocabulary to improve reading ease."
    elif 30 <= reading_ease < 50:
        ease_description = "Difficult to read."
        ease_suggestion = "Use simpler words and shorter sentences, and add headings and subheadings to break up long text blocks."
    else:
        ease_description = "Very difficult to read, often technical or academic."
        ease_suggestion = "Consider rewriting the content with simpler language, more examples, and better formatting for clarity."
        
    # If there are no suggestions set, then change the suggestion to 'No Suggestions'
    if not grade_suggestion:
        grade_suggestion = "No Suggestions"
    if not ease_suggestion:
        ease_suggestion = "No Suggestions"

    return grade_description, ease_description, grade_suggestion, ease_suggestion

#Extract Keywords
def extract_keywords_from_content(content, top_n=20):
    """Extracts keywords from blog content using NLP."""
    try:
        # Print the text before sending to LLM
        cleaned_content = clean_placeholder_text(content)
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
            template="""Analyze the following content based on SEO keyword optimization guidelines. Provide a detailed, accurate analysis, evaluation of how well the content adheres to each guideline. In addition, provide suggestions to improve the page in terms of keyword optimization for SEO Performance. If there are no suggestions for improvement, then explicitly state 'No Suggestions'. Do not include any concluding statements or summaries.

        Format the analysis and suggestions as follows, with each output on a separate line:

        Keyword and Search Intent Alignment:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Primary Keyword in Page Title:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Page Title Engagement:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Page Title Modifiers:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Page Title Character Length:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Page Title HTML Structure:  [Your analysis here. If you can't assess, state that]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Primary Keyword in Meta Description:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Primary Keyword in URL:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Primary Keyword in First Sentence:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Keyword Density:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Top 5 Keywords Distribution:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']
        Variations and LSI Keywords:  [Your analysis here]. Suggestions: [Specific suggestions here or 'No Suggestions']

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
            Assess whether the page title could benefit from the inclusion of a temporal modifier, such as a year (e.g., '2025'), considering its current click-worthiness and character length.
            Confirm whether the page title utilizes the maximum character length without exceeding it while remaining clear and informative.
            Verify that the page title is wrapped in an H1 tag and follows correct HTML structure.
            Evaluate the inclusion and usage of the primary keyword in the meta description and its effectiveness in compelling users.
            Determine if the primary keyword is present in the URL, and assess whether the URL structure is lean and optimized for SEO.
            Analyze the placement of the primary keyword in the content, especially in the first sentence.
            Assess the keyword density, ensuring it is balanced and consistent with competitor content.
            Ensure that top 5 keywords are distributed across various article sections.
            Evaluate the use of variations of the primary keyword and synonyms (LSI keywords) throughout the content for effective optimization.
            
        The evaluation should deliver a professional, high-quality response that adheres to these standards.
    """
)

        cleaned_content = clean_placeholder_text(content, url)
        cleaned_page_title = clean_placeholder_text(page_title)
        cleaned_meta_description = clean_placeholder_text(meta_description)
        cleaned_url = clean_placeholder_text(url)
        # Print the cleaned text before sending to LLM
        print_text_before_llm(f"Content: {cleaned_content}, Page Title: {cleaned_page_title}, Meta Description: {cleaned_meta_description}, URL: {cleaned_url}", "Text Before SEO Optimization LLM:")

        response = llm.invoke(prompt.format(
            content=cleaned_content,
            page_title=cleaned_page_title,
            meta_description=cleaned_meta_description,
            url=cleaned_url,
        ))
        # Process and return the detailed evaluation as a list
        processed_response = []
        for line in response.content.strip().split("\n"):
          parts = line.split("Suggestions:")
          if len(parts) == 2:
            analysis, suggestions = parts
            if not suggestions.strip():
                processed_response.append(f"{analysis.strip()} Suggestions: No Suggestions")
            else:
              processed_response.append(line)
          else:
             processed_response.append(line)
        return [remove_zw_chars(line) for line in processed_response]
    except Exception as e:
        st.error(f"Error optimizing SEO keywords: {e}")
        log_error("Error in optimize_seo_keywords", e)
        return []

#Evaluate Content quality of content
def evaluate_content_quality(content, llm):
    """Evaluates content quality based on structured guidelines using an LLM, and also provides suggestions."""
    try:
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""Analyze the following content based on content quality guidelines. Provide a detailed,accurate analysis, fact-based evaluation of how well the content adheres to each guideline. In addition, provide specific, actionable suggestions to improve the content quality *specifically for SEO performance and user engagement*. If there are no specific, actionable suggestions for SEO performance or user engagement improvements based on the available information, then explicitly state 'No Suggestions'. Do not include any concluding statements or summaries.

    Format the analysis and suggestions as follows, with each output on a separate line:

    Spelling and Grammar: [Your analysis here, state clearly if any spacing, spelling or grammatical issues are present]. Suggestions: [Specific suggestions for SEO performance and user engagement here or 'No Suggestions']
    Scannability:  [Your analysis here, describe how headings, bullet points, and other elements are used to make the content easy to scan. Provide examples of where elements make it easy to scan the text]. Suggestions: [Specific suggestions for SEO performance and user engagement here or 'No Suggestions']
    Readability:  [Your analysis here, highlight specific sentences or sections that are overly complex and assess if the content meets an 8th-grade readability level]. Suggestions: [Specific suggestions for SEO performance and user engagement here or 'No Suggestions']
    Engagement:  [Your analysis here, assess how the content effectively captures and maintains the reader's attention, and make note of any specific engaging or disengaging elements]. Suggestions: [Specific suggestions for SEO performance and user engagement here or 'No Suggestions']
    Paragraph Structure:  [Your analysis here, check if paragraphs are short, concise and structured appropriately. Highlight if any sections use dense blocks of text]. Suggestions: [Specific suggestions for SEO performance and user engagement here or 'No Suggestions']
    Heading Structure:  [Your analysis here, provide an analysis of the heading structure and if it helps the reader navigate the content and its logical flow]. Suggestions: [Specific suggestions for SEO performance and user engagement here or 'No Suggestions']
    Heading Clarity:  [Your analysis here, assess if headings are descriptive, clear and accurately reflect the topic of each section, provide details]. Suggestions: [Specific suggestions for SEO performance and user engagement here or 'No Suggestions']
    Keyword Usage:  [Your analysis here, assess the use of keyword variations, LSI keywords, or synonyms in the headings and throughout the content and note the relevance and frequency of their usage]. Suggestions: [Specific suggestions for SEO performance and user engagement here or 'No Suggestions']
    Use of Lists: [Your analysis here, confirm if bullet points and numbered lists are used effectively and enhance clarity and structure]. Suggestions: [Specific suggestions for SEO performance and user engagement here or 'No Suggestions']
    Originality and Relevance: [Your analysis here, validate the originality and relevance of the content. State if it aligns with current trends and provides up-to-date information. Also comment if the content appears to be well-researched and informative]. Suggestions: [Specific suggestions for SEO performance and user engagement here or 'No Suggestions']


    Blog Content:
    {content}

    Content Quality Guidelines:
        Examine the content to check for spacing, spelling and grammatical errors using tools like Grammarly. Clearly state whether any issues were identified.
        Assess the content's readability and formatting. Confirm if headings, bullet points, or other elements make the content easy to scan and consume.
        Ensure the content is written at an 8th-grade readability level. Highlight any sentences or sections that are overly complex.
        Evaluate whether the content effectively captures and maintains the reader's attention throughout. Indicate any sections that might lack engagement.
        Verify that paragraphs are short and structured to avoid dense blocks of text. Mention if any sections deviate from this guideline.
        Analyze the logical flow of the headings. Confirm whether they guide the reader effectively through the content.
        Check if the headings are descriptive and accurately reflect the topic of each section.
        Evaluate the use of keyword variations, LSI keywords, or synonyms in the headings and throughout the content. Note the relevance and frequency of their usage.
        Verify the use of bullet points and numbered lists where applicable. Confirm whether they enhance clarity and structure.
        Validate the originality and relevance of the content. State whether it aligns with current trends and provides up-to-date information.

    The evaluation should deliver a professional, high-quality response that adheres to these standards.
    """
)

        # Clean the text *before* printing and sending to LLM
        cleaned_content = clean_placeholder_text(content)
        print_text_before_llm(cleaned_content, "Text Before Content Quality LLM:")

        response = llm.invoke(prompt.format(content = cleaned_content))
       # Process and return the detailed evaluation as a list
        processed_response = []
        for line in response.content.strip().split("\n"):
           parts = line.split("Suggestions:")
           if len(parts) == 2:
               analysis, suggestions = parts
               if not suggestions.strip():
                   processed_response.append(f"{analysis.strip()} Suggestions: No Suggestions")
               else:
                   processed_response.append(line)
           else:
               processed_response.append(line)
        return [remove_zw_chars(line) for line in processed_response]
    except Exception as e:
        st.error(f"Error evaluating content quality: {e}")
        log_error("Error in evaluate_content_quality", e)
        return []

prompt_template = PromptTemplate.from_template("""
Analyze the following page content and evaluate its link structure according to the guidelines provided. Provide a detailed,accurate analysis, fact-based evaluation of how well the content adheres to each guideline, focusing on concrete HTML elements and patterns rather than inferences. Include examples of the detected elements from the page content where possible. In addition, provide specific, actionable suggestions to improve the link structure *specifically for SEO performance*. If there are no specific, actionable suggestions for SEO improvement based on the available information, then explicitly state 'No Suggestions'. Do not include any concluding statements.

Format the analysis and suggestions as follows, with each output on a separate line:

Internal Links:  Based on the provided page content, identify if internal links (links to other pages on the same website) are present and provide details on where they are located and the types of links they appear to be (e.g. links in the body, navigation or footer). If found, state where the internal links are located and what type of links they are, otherwise state 'No Internal Links Found'.  Suggestions: [Specific suggestions for SEO performance here or 'No Suggestions']
Descriptive Anchor Text:  Based on the provided page content, analyze if the internal links use descriptive anchor text (text that clearly indicates the target page's content) and provide examples. State if descriptive anchor text is used and provide examples, based on what can be determined from the content. Suggestions: [Specific suggestions for SEO performance here or 'No Suggestions']
Internal Link Optimization: Analyze the page content. Based on what can be determined from the content, are important internal links (e.g. higher value or more relevant pages) placed higher or earlier in the page or are they not prioritized? Identify which links appear to be more important based on their placement. State if important links are prioritized, or if they are not and provide details.  Suggestions: [Specific suggestions for SEO performance here or 'No Suggestions']
Breadcrumbs: Analyze the page and ensure Breadcrumbs are present or not, at least including "Home" and "Blog". The exact structure would need confirmation by inspecting the HTML. Suggestions: [Specific suggestions for SEO performance here or 'No Suggestions']
Usefulness of Internal Links: Based on the page content, assess if internal links are useful for the user (e.g., do they link to related content that helps the user). Provide details of what kind of links they are linking to, and state if they are useful or not, based on what can be determined from the content. Suggestions: [Specific suggestions for SEO performance here or 'No Suggestions']
Preferred URLs for Internal Links: Based on the page content, analyze the URLs of the internal links. Are they using the preferred or canonical versions (e.g., ensuring all links to the same page are using the same URL format)? Provide examples, stating if this can be determined or if not. Suggestions: [Specific suggestions for SEO performance here or 'No Suggestions']
External Links: Based on the page content, determine if the page includes external links (links to other websites) to relevant sources, partners, or content, and state if this is highly likely or not. Provide details and why this determination has been made based on what is present in the content. Suggestions: [Specific suggestions for SEO performance here or 'No Suggestions']
Affiliate and Sponsored Links: Based on the page content, determine if there are affiliate, sponsored, or paid external links on the page. Based on the context, determine how likely it is that these exist. State if it can be inferred or not. If it can be inferred that these exist, make an inference if they are *likely* to be 'nofollow'. State if this is an inference from the provided content, or if it cannot be determined. Suggestions: [Specific suggestions for SEO performance here or 'No Suggestions']
External Links Opening in New Window: Based on the page content, and using best practices, are external links *likely* set to open in a new window? State if this is an inference from the provided content, or if it cannot be determined. Suggestions: [Specific suggestions for SEO performance here or 'No Suggestions']
Broken Links: Based on the page content, can it be *reasonably* inferred that there might be any broken links (either internal or external) on the page? Provide your analysis and explain why, and if this is not something that is possible to ascertain from the content, say that. Suggestions: [Specific suggestions for SEO performance here or 'No Suggestions']


Page Content:
{content}
""")

def analyze_url(soup, llm):
    """Sends a prompt to the LLM and returns the analysis."""
    text_content = soup.get_text(separator=" ", strip=True)
    prompt = prompt_template.format(content=clean_placeholder_text(text_content))
    response = llm.invoke(prompt)
    
    processed_response = []
    for line in response.content.strip().split("\n"):
        if ": " in line: # Check for both colon and space
            parts = line.split(": ", 1)  # Split at the first occurrence of ": "
            if len(parts) == 2:
                label, text = parts
                if "Suggestions" in label:
                   if not text.strip():
                     processed_response.append(f"{label.strip()}: No Suggestions")
                   else:
                     processed_response.append(f"{label.strip()}: {text.strip()}")
                else:
                     processed_response.append(f"{label.strip()}: {text.strip()}")
            else:
              processed_response.append(line)
        else:
            processed_response.append(line)

    return [remove_zw_chars(line) for line in processed_response]

# Streamlit App
import streamlit as st

def main():
    st.title("Blog SEO Analyzer")
    st.markdown("---")
    
    blog_url = st.text_input("Enter Blog URL:")
    analyze_btn = st.button("Analyze")
    suggest_btn = st.button("Show Suggestions")
    
    if analyze_btn or suggest_btn:
        with st.spinner("Processing..."):
            soup = scrape_page_content(blog_url)
            if not soup:
                st.error("Failed to retrieve page content")
                return
            
            content, title, meta_description = retrieve_blog_content(blog_url, soup)
            if not content:
                st.error("Failed to extract content from the blog")
                return

            if analyze_btn:
                show_analysis(content, title, meta_description, soup, blog_url)
            if suggest_btn:
                show_suggestions(content, title, meta_description, soup, blog_url)

def show_analysis(content, title, meta_description, soup, blog_url):
    st.subheader("Analysis")
    readability_grade, readability_ease = calculate_readability(content)
    grade_description, ease_description, _, _ = describe_readability(readability_grade, readability_ease)
    
    with st.expander("Readability Scores"):
        st.write(f"**Flesch-Kincaid Grade Level:** {readability_grade} - {grade_description}")
        st.write(f"**Flesch Reading Ease:** {readability_ease} - {ease_description}")
    
    extracted_keywords = extract_keywords_from_content(content)
    with st.expander("Extracted Keywords"):
        st.write("\n".join(extracted_keywords) if extracted_keywords else "No keywords found.")
    
    seo_analysis = optimize_seo_keywords(content, title, meta_description, blog_url, llm)
    with st.expander("Keyword Optimization Analysis"):
        for item in seo_analysis:
            analysis, _ = item.split("Suggestions:") if "Suggestions:" in item else (item, "")
            st.write(analysis.strip())
    
    content_quality = evaluate_content_quality(content, llm)
    with st.expander("Content Evaluation Analysis"):
        for item in content_quality:
            analysis, _ = item.split("Suggestions:") if "Suggestions:" in item else (item, "")
            st.write(analysis.strip())
    
    link_analysis = analyze_url(soup, llm)
    with st.expander("Link Evaluation"):
        for item in link_analysis:
            analysis, _ = item.split("Suggestions:") if "Suggestions:" in item else (item, "")
            st.write(analysis.strip())

def show_suggestions(content, title, meta_description, soup, blog_url):
    st.subheader("Suggestions")
    
    _, _, grade_suggestion, ease_suggestion = describe_readability(*calculate_readability(content))
    if grade_suggestion or ease_suggestion:
        with st.expander("Readability Suggestions"):
            if grade_suggestion:
                st.write(f"- {grade_suggestion}")
            if ease_suggestion:
                st.write(f"- {ease_suggestion}")
    
    seo_suggestions = optimize_seo_keywords(content, title, meta_description, blog_url, llm)
    if any("Suggestions:" in item and "No Suggestions" not in item for item in seo_suggestions):
        with st.expander("Keyword Optimization Suggestions"):
            for item in seo_suggestions:
                _, suggestions = item.split("Suggestions:") if "Suggestions:" in item else ("", "")
                if suggestions.strip() and suggestions.strip() != "No Suggestions":
                    st.write(suggestions.strip())
    
    content_suggestions = evaluate_content_quality(content, llm)
    if any("Suggestions:" in item and "No Suggestions" not in item for item in content_suggestions):
        with st.expander("Content Evaluation Suggestions"):
            for item in content_suggestions:
                _, suggestions = item.split("Suggestions:") if "Suggestions:" in item else ("", "")
                if suggestions.strip() and suggestions.strip() != "No Suggestions":
                    st.write(suggestions.strip())
    
    link_suggestions = analyze_url(soup, llm)
    if any("Suggestions:" in item and "No Suggestions" not in item for item in link_suggestions):
        with st.expander("Link Evaluation Suggestions"):
            for item in link_suggestions:
                _, suggestions = item.split("Suggestions:") if "Suggestions:" in item else ("", "")
                if suggestions.strip() and suggestions.strip() != "No Suggestions":
                    st.write(suggestions.strip())

if __name__ == "__main__":
    main()
