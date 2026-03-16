from bs4 import BeautifulSoup

# dummy function for future website parsing
def parse_website_content(html):
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("http"):
            urls.append((href, a_tag))
        elif href.startswith("//"):
            urls.append(("https:" + href, a_tag))


    return {
        'urls' : urls
    }