import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

def analyze_links(url):
    try:
        # Fetch the HTML content of the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Exclude the footer section
        if soup.find('footer'):
            footer = soup.find('footer')
            footer.extract()
        
        # Extract all links
        links = soup.find_all('a', href=True)
        internal_links = []
        external_links = []
        broken_links = []
        
        base_url = "{0.scheme}://{0.netloc}".format(urlparse(url))
        
        for link in links:
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Categorize internal and external links
            if base_url in full_url:
                internal_links.append((full_url, link.text.strip()))
            else:
                external_links.append((full_url, link.text.strip()))
            
            # Check for broken links
            try:
                link_response = requests.head(full_url, timeout=5)
                if link_response.status_code >= 400:
                    broken_links.append(full_url)
            except requests.RequestException:
                broken_links.append(full_url)
        
        # Analyze internal links
        internal_link_details = {
            "internal_links_count": len(internal_links),
            "descriptive_anchor_texts": [text for url, text in internal_links if text],
            "internal_links": internal_links
        }
        
        # Analyze external links
        external_link_details = {
            "external_links_count": len(external_links),
            "nofollow_tags": [url for url, text in external_links if 'rel="nofollow"' in str(link)],
            "new_window": [url for url, text in external_links if 'target="_blank"' in str(link)],
            "broken_links": broken_links
        }
        
        # Breadcrumbs
        breadcrumbs = bool(soup.find("nav", {"aria-label": "breadcrumb"}))
        
        # Generate the report
        return {
            "internal_links": internal_link_details,
            "external_links": external_link_details,
            "breadcrumbs": breadcrumbs,
        }
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error processing content: {e}"

# URL of the blog to analyze
url = "https://www.infisign.ai/blog/top-10-single-sign-on-sso-providers-solutions-in-2024"
analysis_result = analyze_links(url)

# Print the results
print(analysis_result)
