import requests
from bs4 import BeautifulSoup

# Fetch the webpage
url = "https://www.infisign.ai/blog/top-10-single-sign-on-sso-providers-solutions-in-2024"
response = requests.get(url)

if response.status_code == 200:
    # Parse the HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the main content area (adjust the selector as needed for the specific site)
    main_content = soup.find("article")  # or soup.find("div", class_="main-content") based on the website structure

    if main_content:
        # Get the text inside the main content
        clean_text = main_content.get_text(separator="\n", strip=True)

        # Print cleaned text
        print("Cleaned Text Content:\n")
        print(clean_text[:1000])  # Preview first 1000 characters

        # Save cleaned text to a file
        with open("cleaned_blog_content.txt", "w", encoding="utf-8") as file:
            file.write(clean_text)
    else:
        print("Main content area not found.")
else:
    print(f"Failed to fetch the webpage. Status code: {response.status_code}")
