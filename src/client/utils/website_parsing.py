from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit
from sklearn.feature_extraction.text import CountVectorizer
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# dummy function for future website parsing
def parse_website_content(html, url):
    # soup = BeautifulSoup(html, "html.parser")
    # urls = []

    # for a_tag in soup.find_all("a", href=True):
    #     href = a_tag["href"]
    #     if href.startswith("http"):
    #         urls.append((href, a_tag))
    #     elif href.startswith("//"):
    #         urls.append(("https:" + href, a_tag))


    # return {
    #     'urls' : urls
    # }
    return extract_urls(html, url)


def get_topic_from_url(url):
    url_parts = get_url_parts(url)
    try:
        topic = url_parts[2]
    except:
        return None
    return topic

def get_url_parts(url, delim="/"):
    return urlsplit(url).path.split(delim)

def extract_urls(html, url):
    bs = BeautifulSoup(html)
    metadata_dict = dict()
    #print(bs.prettify())
    links = bs.find_all("a", href=True)
    print(f"Num of all links: {len(links)}")
    url_parts = get_url_parts(url)
    for l in links:
        key = urljoin(url, l["href"])
        key_parts = get_url_parts(key)
        #print(url_parts)
        if len(key_parts) < 2:
            #print("URL parts too short")
            continue
        if key_parts[1] == "s":
            continue
        if "24ur.com" not in key:
            continue
        metadata_dict[key] = dict({
            "source": url,
            "source_title": url_parts[-1] if ".html" in url else None,
            "section": key_parts[1] if len(key_parts) > 1 else None,
            "topic": key_parts[2] if len(key_parts) > 2 else None,
            "container_id": None
        })

        if key_parts[1] == "kljucna-beseda":
            metadata_dict[key]["container_id"] = "kljucna-beseda"

        if key_parts[1] == "spored":
            metadata_dict[key]["container_id"] = "spored"
        # if key_parts[1] == "vreme":
        #     print("Vreme")

        parent_3 = l.parent.parent.parent
        parent_2 = l.parent.parent
        parent_1 = l.parent

        if parent_3.has_attr("class"):
            if "menu__items" in parent_3.attrs["class"]:
                metadata_dict[key]["container_id"] = "menu"
            elif "submenu" in parent_3.attrs["class"]:
                metadata_dict[key]["container_id"] = "submenu"
        # print(parent_2.attrs)
        if parent_2.has_attr("id"):
            if parent_2.attrs["id"] == "footer-bottom":
                metadata_dict[key]["container_id"] = "footer"
            


    # related articles
    related_articles = bs.find(id="related-articles")
    if related_articles != None:
        related_links = related_articles.find_all("a", href=True)
        print("Related articles:")
        for l in related_links:
            key = urljoin(url, l["href"])
            if key not in metadata_dict:
                continue
            metadata_dict[key]["container_id"] = "related-articles"
            print(key)

    # Read more links
    read_mores = bs.find_all("a", class_="read-more")
    print("Read mores:")
    for l in read_mores:
        key = urljoin(url, l["href"])
        if key not in metadata_dict:
            continue
        metadata_dict[key]["container_id"] = "read-more"
        print(key)

    # Proad recommends
    print("Proad links: ")
    proads_div = bs.find(id="proad")
    if proads_div != None:
        proads_links = proads_div.find_all("a", href=True)
        for l in proads_links:
            if l["href"] == "/vsebine/oglasevanje":
                continue
            key = urljoin(url, l["href"])
            if key not in metadata_dict:
                continue
            metadata_dict[key]["container_id"] = "proad"
            print(key)
    return metadata_dict
