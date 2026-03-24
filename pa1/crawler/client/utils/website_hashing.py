import hashlib
import re
from bs4 import BeautifulSoup


def hash_website(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = text[:10000]

    return hashlib.sha256(text.encode("utf-8")).hexdigest()