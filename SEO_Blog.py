# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse
# import re
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.schema.runnable import RunnableSequence
# import textstat
# import plotly.express as px
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# # Step 1: Retrieve Blog Content
# def retrieve_blog_content(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     content = " ".join([p.text for p in soup.find_all('p')])
#     title = soup.title.string if soup.title else "No title found"
    
#     # Extract meta description
#     meta_description = (
#         soup.find("meta", {"name": "description"}) or
#         soup.find("meta", {"property": "og:description"})
#     )
#     meta_description = meta_description["content"] if meta_description else "No meta description found"
    
#     return content, title, meta_description

# # Step 2: Extract Domain and Subdomain
# def extract_domain_subdomain(url):
#     parsed_url = urlparse(url)
#     domain_parts = parsed_url.netloc.split('.')
#     domain = ".".join(domain_parts[-2:])
#     subdomain = parsed_url.netloc if len(domain_parts) > 2 else "No subdomain"
#     return domain, subdomain

# # Step 3: Readability Score
# def readability_score(content):
#     kincaid_grade = textstat.flesch_kincaid_grade(content)
#     reading_ease = textstat.flesch_reading_ease(content)
#     return kincaid_grade, reading_ease

# # Step 3: Suggest SEO Keywords using LLM
# def suggest_keywords_with_llm(content):
#     llm = ChatGoogleGenerativeAI(api_key="AIzaSyDR0Sr1VV1TJl3AFNScIdubB7JkyUsJhSo",model="gemini-1.5-pro-002",temperature=0.3)
#     prompt_template = PromptTemplate(
#         input_variables=["content"],
#         template="""
#         Analyze the following blog content and generate a list of highly relevant SEO keywords. Focus on terms with high search intent and prioritize short-tail keywords,intermediate-tail keywords.\n\nBlog Content:\n{content}
#         """
#     )
#     sequence = prompt_template | llm  # Chain the prompt and LLM together
#     response = sequence.invoke({"content": content})
#     keywords = response.content.strip().split(",")  # Split into keywords by comma
#     keywords = [keyword.strip() for keyword in keywords if keyword.strip()]  # Remove extra spaces and empty keywords
    
#     return keywords
# # Step 4: Provide Suggestions Based on Keywords and Blog Content
# def provide_suggestions(blog_content, title, meta_description, keywords):

#     keyword_string = ", ".join(keywords)
#     llm = ChatGoogleGenerativeAI(api_key="AIzaSyDR0Sr1VV1TJl3AFNScIdubB7JkyUsJhSo",model="gemini-1.5-pro-002",temperature=0.3)
#     prompt_template = PromptTemplate(
#         input_variables=["content","title", "meta_description", "keywords"],
#         template = """

# Based on the following blog content , title, meta description and SEO keywords, provide actionable suggestions to improve the blog's SEO performance:

# 1. **Title**  must be (50-55 characters) 
#    - Analyze the current title: {title}
#    - Suggest improvements if necessary.

# 2. **Meta Description**  must be(150-155 characters)
#    - Analyze the current meta description: {meta_description}
#    - Suggest improvements if necessary.

# 3. **Content**:
#    - Improve keyword usage and structure (headings, subheadings, internal links).
#    - Ensure readability and keyword density.
#    - Suggest updates for any outdated content (e.g., years, trends).

# Blog Content:\n{content}\n\n
# SEO Keywords:\n{keywords}
# """

#     )
#     sequence = prompt_template | llm  # Chain the prompt and LLM together
#     response = sequence.invoke({"content": blog_content, "title":title, "meta_description": meta_description, "keywords": keyword_string}) 
#     return response.content.strip()

# def compare_with_competitors(user_keywords, competitor_content):
#     llm = ChatGoogleGenerativeAI(api_key="AIzaSyDR0Sr1VV1TJl3AFNScIdubB7JkyUsJhSo", model="gemini-1.5-pro-002", temperature=0.3)
#     prompt_template = PromptTemplate(
#         input_variables=["user_keywords", "competitor_content"],
#         template="""
#         Compare the following SEO keywords with the content of a competitor's blog. Identify gaps where relevant keywords from the user's list are missing in the competitor's content. Provide actionable insights for improvement.\n\nUser Keywords:\n{user_keywords}\n\nCompetitor Content:\n{competitor_content}
#         """
#     )
#     sequence = prompt_template | llm
#     response = sequence.invoke({
#         "user_keywords": ", ".join(user_keywords),
#         "competitor_content": competitor_content
#     })
#     return response.content.strip()


# def visualize_word_cloud(keywords):
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(keywords))
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     st.pyplot(plt)


# # Streamlit UI
# st.title("Blog SEO Analyzer")

# # Input blog URL
# blog_url = st.text_input("Enter Blog URL:")
# submit_button = st.button("Submit")

# if submit_button and blog_url:
#     with st.spinner("Retrieving blog content..."):
#         try:
#             # Step 1: Retrieve blog content
#             blog_content = retrieve_blog_content(blog_url)
#             blog_content, title, meta_description = retrieve_blog_content(blog_url)
#             domain, subdomain = extract_domain_subdomain(blog_url)
            
#             readability = readability_score(blog_content)
            
#             st.subheader("Page Information")
#             st.write(f"**Title:** {title}")
#             st.write(f"**Meta Description:** {meta_description}")

#             st.subheader("Blog Information")
#             st.write(f"**Domain:** {domain}")
#             st.write(f"**Subdomain:** {subdomain}")

#             st.subheader("Readability Score")
#             col1, col2 = st.columns(2)

#             # Display Flesch-Kincaid Grade in the first column
#             with col1:
#                st.metric(label="Flesch-Kincaid Grade", value=f"{readability[0]:.2f}")

#             # Display Flesch Reading Ease in the second column
#             with col2:
#                st.metric(label="Flesch Reading Ease", value=f"{readability[1]:.2f}")
            
#             # Step 2: Suggest Keywords
#             st.subheader("Suggested Keywords")
#             keywords = suggest_keywords_with_llm(blog_content)
#             st.write(", ".join(keywords))
            
            
#             # Step 3: Provide SEO Suggestions
#             st.subheader("SEO Suggestions")
#             seo_suggestions = provide_suggestions(blog_content,title,meta_description, keywords)
#             st.text(seo_suggestions)

#             st.subheader("Keyword Word Cloud")
#             visualize_word_cloud(keywords)

            

            
#             # Competitor blog URL input
#             competitor_url = st.text_input("Enter Competitor Blog URL:")
#             if competitor_url:
#                 with st.spinner("Retrieving competitor blog content..."):
#                     try:
#                         # Retrieve competitor blog content
#                         competitor_content = retrieve_blog_content(competitor_url)

#                         # Compare user keywords with competitor content
#                         st.subheader("Competitor Analysis")
#                         comparison_results = compare_with_competitors(keywords, competitor_content)
#                         st.text(comparison_results)

#                     except Exception as e:
#                         st.error(f"Error retrieving competitor content: {e}")
#         except Exception as e:
#             st.error(f"Error retrieving or processing blog content: {e}")


import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import textstat
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Constants
API_KEY = "AIzaSyDR0Sr1VV1TJl3AFNScIdubB7JkyUsJhSo"  # Replace with your API key
LLM_MODEL = "gemini-1.5-pro-002"
LLM_TEMPERATURE = 0.3

# Initialize LLM
llm = ChatGoogleGenerativeAI(api_key=API_KEY, model=LLM_MODEL, temperature=LLM_TEMPERATURE)

# Step 1: Retrieve Blog Content
def retrieve_blog_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = " ".join([p.text for p in soup.find_all('p')])
        title = soup.title.string if soup.title else "No title found"

        meta_tag = soup.find("meta", {"name": "description"}) or soup.find("meta", {"property": "og:description"})
        meta_description = meta_tag["content"] if meta_tag else "No meta description found"
        
        return content, title, meta_description
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving blog content: {e}")
        return None, None, None

# Step 2: Extract Domain and Subdomain
def extract_domain_subdomain(url):
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    domain = ".".join(domain_parts[-2:])
    subdomain = parsed_url.netloc if len(domain_parts) > 2 else "No subdomain"
    return domain, subdomain

# Step 3: Calculate Readability Score
def calculate_readability(content):
    if not content:
        return None, None
    kincaid_grade = textstat.flesch_kincaid_grade(content)
    reading_ease = textstat.flesch_reading_ease(content)
    return kincaid_grade, reading_ease

# Step 4: Suggest SEO Keywords
def suggest_seo_keywords(content):
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""
         Analyze the following blog content and generate a list of highly relevant SEO keywords. Focus on terms with high search intent and prioritize short-tail keywords,intermediate-tail keywords.
        
        Blog Content:
        {content}
        """
    )
    response = (prompt | llm).invoke({"content": content})
    keywords = response.content.strip().split(",")
    return [kw.strip() for kw in keywords if kw.strip()]

# Step 5: Provide SEO Suggestions
def generate_seo_suggestions(content, title, meta_description, keywords):
    prompt = PromptTemplate(
        input_variables=["content", "title", "meta_description", "keywords"],
        template="""
        Based on the following blog content, title, meta description and SEO keywords, provide actionable suggestions to improve the blog's SEO performance:

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

# Step 6: Visualize Keywords as Word Cloud
def visualize_word_cloud(keywords):
    if not keywords:
        st.warning("No keywords to display.")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(keywords))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# Streamlit App
def main():
    st.title("Blog SEO Analyzer")

    # Input blog URL
    blog_url = st.text_input("Enter Blog URL:")
    if st.button("Analyze Blog") and blog_url:
        with st.spinner("Processing..."):
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
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Flesch-Kincaid Grade", f"{readability[0]:.2f}")
            with col2:
                st.metric("Flesch Reading Ease", f"{readability[1]:.2f}")

            st.subheader("Suggested Keywords")
            st.write(", ".join(keywords))

            st.subheader("SEO Suggestions")
            seo_suggestions = generate_seo_suggestions(content, title, meta_description, keywords)
            st.text(seo_suggestions)

            st.subheader("Keyword Word Cloud")
            visualize_word_cloud(keywords)

if __name__ == "__main__":
    main()