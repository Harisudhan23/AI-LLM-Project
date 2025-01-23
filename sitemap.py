import requests
from bs4 import BeautifulSoup

def fetch_sitemap(url):
    try:
        # Send a request to the sitemap URL
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"Sitemap found at: {url}")
            return response.text
        else:
            print(f"Sitemap not found at {url}. HTTP Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return None

def parse_sitemap(sitemap_xml):
    # Parse the XML to extract URLs
    soup = BeautifulSoup(sitemap_xml, "xml")
    urls = [loc.text for loc in soup.find_all("loc")]
    print(f"Found {len(urls)} URLs in the sitemap.")
    return urls

# Step 1: Try standard sitemap URL
sitemap_url = "https://www.infisign.ai/blog/top-10-single-sign-on-sso-providers-solutions-in-2024"
sitemap_content = fetch_sitemap(sitemap_url)

# Step 2: Parse the sitemap if found
if sitemap_content:
    urls = parse_sitemap(sitemap_content)
    print("\nFirst 5 URLs in the sitemap:")
    print("\n".join(urls[:5]))
else:
    print("No sitemap content found.")
