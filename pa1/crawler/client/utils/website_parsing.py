from datetime import datetime
import hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit
from sklearn.feature_extraction.text import CountVectorizer
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from utils.url_cleaning import normalize_url, canonicalize_url # type: ignore
from utils.page_data_objects import PageDbSaveObject
from utils.website_hashing import hash_website # type: ignore

BINARY_FILE_TYPES = {"PDF", "DOC", "DOCX", "PPT", "PPTX"}
BINARY_FILE_EXTENSIONS = (".pdf", ".doc", ".docx", ".ppt", ".pptx")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")

def parse_website_content(html, url, robots):
    
    urls_dict = extract_urls(html, url)
    for key in list(urls_dict.keys()):
        if not robots.is_allowed(robots.user_agent, key):
            del urls_dict[key]

    return urls_dict


def get_topic_from_url(url):
    url_parts = get_url_parts(url)
    try:
        topic = url_parts[2]
    except:
        return None
    return topic

def get_url_parts(url, delim="/"):
    return urlsplit(url).path.split(delim)

def append_data(dictionary, key, new_metadata):
    # assuming key already exists in dictionary
    dictionary[key]["container_id"].extend(new_metadata["container_id"])

def extract_urls(html, url):

    # add url cleaning

    bs = BeautifulSoup(html, "html.parser")
    metadata_dict = dict()
    #print(bs.prettify())
    links = bs.find_all("a", href=True)
    print(f"Num of all links: {len(links)}")

    # extract all article keywords
    article_keywords = []
    keyword_links_div = bs.find(id="article-keywords")
    if keyword_links_div is not None:
        keyword_links = keyword_links_div.find_all("a")
        for keyword in keyword_links:
            article_keywords.append(keyword.text.rstrip())
    
    # print(article_keywords)

    url_parts = get_url_parts(url)
    for l in links:
        key = normalize_url(urljoin(url, l["href"]))
        key_parts = get_url_parts(key)

        # find summary text
        summary_p = l.find(["p", "span"], class_="summary")
        
        # find label text
        label_span = l.find("span", class_="label")

        #print(url_parts)
        if len(key_parts) < 2:
            #print("URL parts too short")
            continue
        if key_parts[1] == "s":
            continue
        if "24ur.com" not in key:
            continue
        
        new_metadata = dict({
            "source": url,
            "source_title": url_parts[-1] if ".html" in url else None,
            "section": key_parts[1].replace("-", " ") if len(key_parts) > 1 else None,
            "topic": key_parts[2].replace("-", " ") if len(key_parts) > 2 else None,
            "link_title": key_parts[-1].replace("-", " ")[:-5] if ".html" in key else "",
            "summary": summary_p.text if summary_p != None else None,
            "container_id": [],
            "article_keywords":  article_keywords,
            "link_keywords": label_span.text if label_span != None else None
        })


        if key_parts[1] == "kljucna-beseda":
            new_metadata["container_id"].append("kljucna-beseda")
        elif key_parts[1] == "spored":
            new_metadata["container_id"].append("spored")

        parent_3 = l.parent.parent.parent
        parent_2 = l.parent.parent
        parent_1 = l.parent

        if parent_3.has_attr("class"):
            if "menu__items" in parent_3.attrs["class"]:
                new_metadata["container_id"].append("menu")
            elif "submenu" in parent_3.attrs["class"]:
                new_metadata["container_id"].append("submenu")
        # print(parent_2.attrs)
        if parent_2.has_attr("id"):
            if parent_2.attrs["id"] == "footer-bottom":
                new_metadata["container_id"].append("footer")
            
        if key in metadata_dict:
            append_data(metadata_dict, key, new_metadata)
        else:
            metadata_dict[key] = new_metadata


    # related articles
    related_articles = bs.find(id="related-articles")
    if related_articles != None:
        related_links = related_articles.find_all("a", href=True)
        # print("Related articles:")
        for l in related_links:
            key = urljoin(url, l["href"])
            if key not in metadata_dict:
                continue
            metadata_dict[key]["container_id"] = "related-articles"
            # print(key)

    # Read more links
    read_mores = bs.find_all("a", class_="read-more")
    # print("Read mores:")
    for l in read_mores:
        key = urljoin(url, l["href"])
        if key not in metadata_dict:
            continue
        metadata_dict[key]["container_id"] = "read-more"
        # print(key)

    # Proad recommends
    # print("Proad links: ")
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
            # print(key)
    return metadata_dict




def get_page_database_save_object(logger, url, html):
    try:
        normalized_url = canonicalize_url(url)
        parsed = urlsplit(normalized_url)
        domain = parsed.netloc

        soup = BeautifulSoup(html, "html.parser")

        if normalized_url.lower().endswith(BINARY_FILE_EXTENSIONS):
            page_type = "BINARY"
            html_content = None
        else:
            page_type = "HTML"
            html_content = html

        page_obj = PageDbSaveObject(
            url=normalized_url,
            site_domain=domain,
            page_type_code=page_type,
            html_content=html_content,
            http_status_code=200,
            accessed_time=datetime.now()
        )

        if html_content:
            page_obj.content_hash = hash_website(html_content, normalized_url)

        seen_links = set()
        for a in soup.find_all("a", href=True):
            href = a.get("href").strip()

            if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue

            absolute_url = urljoin(normalized_url, href)
            clean_url = canonicalize_url(absolute_url)

            if clean_url and clean_url not in seen_links:
                seen_links.add(clean_url)
                page_obj.add_link(clean_url)

        for img in soup.find_all("img"):
            src = img.get("src")

            if not src:
                continue

            src = src.strip()

            img_url = urljoin(normalized_url, src)
            img_url = normalize_url(img_url)

            if not img_url:
                continue

            lower_url = img_url.lower()

            if not lower_url.endswith(IMAGE_EXTENSIONS):
                continue

            filename = img_url.split("/")[-1] or "image"

            page_obj.add_image(
                filename=filename,
                content_type="image",
                data=b"",
                accessed_time=datetime.now()
            )

        if page_type == "BINARY":
            ext = normalized_url.split(".")[-1].upper()

            page_obj.add_page_data(
                data_type_code=ext if ext in BINARY_FILE_TYPES else "PDF",
                data=b""
            )

        return page_obj

    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
        return None