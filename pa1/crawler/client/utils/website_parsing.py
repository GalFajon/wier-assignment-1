from datetime import datetime
import hashlib
import re
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

    # for k in list(urls_dict.keys()):
    #     print(k)


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

def extract_onclick_url(onclick_str):
    if not onclick_str:
        return None

    candidates = []

    patterns = [
        r"(?:location\.href|document\.location|window\.location)\s*=\s*['\"]([^'\"]+)['\"]",
        r"(?:location\.assign|window\.open)\(\s*['\"]([^'\"]+)['\"]"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, onclick_str)
        candidates.extend(matches)

    json_patterns = [
        r'Url\s*:\s*["\']([^"\']+)["\']',
        r'PlaybackUrl\s*:\s*["\']([^"\']+)["\']'
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, onclick_str)
        candidates.extend(matches)

    fallback_matches = re.findall(r'["\']([^"\']+\.html)["\']', onclick_str)
    candidates.extend(fallback_matches)

    return candidates

def process_link(raw_url, l, base_url, url_parts, metadata_dict, article_keywords):
    key = normalize_url(urljoin(base_url, raw_url))
    key_parts = get_url_parts(key)

    if len(key_parts) < 2:
        return
    if key_parts[1] == "s":
        return
    if "24ur.com" not in key:
        return

    summary_p = l.find(["p", "span"], class_="summary")
    label_span = l.find("span", class_="label")

    new_metadata = dict({
        "source": base_url,
        "source_title": url_parts[-1] if ".html" in base_url else None,
        "section": key_parts[1].replace("-", " ") if len(key_parts) > 1 else None,
        "topic": key_parts[2].replace("-", " ") if len(key_parts) > 2 else None,
        "link_title": key_parts[-1].replace("-", " ")[:-5] if ".html" in key else "",
        "summary": summary_p.text if summary_p is not None else None,
        "container_id": [],
        "article_keywords": article_keywords,
        "link_keywords": label_span.text if label_span is not None else None
    })

    if key_parts[1] == "kljucna-beseda":
        new_metadata["container_id"].append("kljucna-beseda")
    elif key_parts[1] == "spored":
        new_metadata["container_id"].append("spored")

    parent_3 = l.parent.parent.parent
    parent_2 = l.parent.parent

    if parent_3.has_attr("class"):
        if "menu__items" in parent_3.attrs["class"]:
            new_metadata["container_id"].append("menu")
        elif "submenu" in parent_3.attrs["class"]:
            new_metadata["container_id"].append("submenu")

    if parent_2.has_attr("id"):
        if parent_2.attrs["id"] == "footer-bottom":
            new_metadata["container_id"].append("footer")

    if key in metadata_dict:
        append_data(metadata_dict, key, new_metadata)
    else:
        metadata_dict[key] = new_metadata


def extract_urls(html, url):
    bs = BeautifulSoup(html, "html.parser")
    metadata_dict = dict()

    # extract article keywords
    article_keywords = []
    keyword_links_div = bs.find(id="article-keywords")
    if keyword_links_div is not None:
        keyword_links = keyword_links_div.find_all("a")
        for keyword in keyword_links:
            article_keywords.append(keyword.text.rstrip())

    url_parts = get_url_parts(url)

    # href links
    links = bs.find_all("a", href=True)
    for l in links:
        process_link(l["href"], l, url, url_parts, metadata_dict, article_keywords)

    # onclick links 
    onclick_links = bs.find_all("a", onclick=True)
    for l in onclick_links:
        extracted_candidates = extract_onclick_url(l["onclick"])
        for extracted in extracted_candidates:
            process_link(extracted, l, url, url_parts, metadata_dict, article_keywords)

    # related articles
    related_articles = bs.find(id="related-articles")
    if related_articles is not None:
        related_links = related_articles.find_all("a", href=True)
        for l in related_links:
            key = urljoin(url, l["href"])
            if key in metadata_dict:
                metadata_dict[key]["container_id"] = "related-articles"

    # read more
    read_mores = bs.find_all("a", class_="read-more")
    for l in read_mores:
        key = urljoin(url, l["href"])
        if key in metadata_dict:
            metadata_dict[key]["container_id"] = "read-more"

    # proad
    proads_div = bs.find(id="proad")
    if proads_div is not None:
        proads_links = proads_div.find_all("a", href=True)
        for l in proads_links:
            if l["href"] == "/vsebine/oglasevanje":
                continue
            key = urljoin(url, l["href"])
            if key in metadata_dict:
                metadata_dict[key]["container_id"] = "proad"

    return metadata_dict



def get_page_database_save_object(logger, url, html):
    try:
        normalized_url = canonicalize_url(url)
        parsed = urlsplit(normalized_url)
        domain = parsed.netloc

        if normalized_url.lower().endswith(BINARY_FILE_EXTENSIONS):
            return PageDbSaveObject(
                url=normalized_url,
                site_domain=domain,
                page_type_code="BINARY",
                html_content=None,
                http_status_code=200,
                accessed_time=datetime.now()
            )

        soup = BeautifulSoup(html, features="html.parser")
        if soup is None:
            logger.error("get_page_database_save_object - SOUP RETURNED A NONE OBJECT")
            return None

        page_obj = PageDbSaveObject(
            url=normalized_url,
            site_domain=domain,
            page_type_code="HTML",
            html_content=html,
            http_status_code=200,
            accessed_time=datetime.now()
        )

        page_obj.content_hash = hash_website(html, normalized_url)

        seen_links = set()
        seen_page_data = set()
        seen_images = set()

        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href:
                continue

            href = href.strip()

            if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue

            absolute_url = urljoin(normalized_url, href)
            clean_url = canonicalize_url(absolute_url)

            if not clean_url:
                continue

            lower_url = clean_url.lower()

            binary_type = None
            for ext in BINARY_FILE_TYPES:
                if lower_url.endswith(f".{ext.lower()}"):
                    binary_type = ext
                    break

            if binary_type:
                if clean_url not in seen_page_data:
                    seen_page_data.add(clean_url)
                    page_obj.add_page_data(
                        data_type_code=binary_type,
                        data=b""
                    )
                continue

            if clean_url not in seen_links:
                seen_links.add(clean_url)
                page_obj.add_link(clean_url)

        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue

            src = src.strip()

            img_url = urljoin(normalized_url, src)
            img_url = canonicalize_url(img_url)

            if not img_url:
                continue

            lower_url = img_url.lower()
            if not lower_url.endswith(IMAGE_EXTENSIONS):
                continue

            if img_url in seen_images:
                continue
            seen_images.add(img_url)

            filename = img_url.split("/")[-1] or "image"

            ext = filename.split(".")[-1].lower() if "." in filename else ""
            
            page_obj.add_image(
                filename=filename,
                content_type=ext,
                data=b"",
                accessed_time=datetime.now()
            )

        return page_obj

    except Exception as e:
        logger.error(f"get_page_database_save_object - Error parsing {url}: {e}")
        return None