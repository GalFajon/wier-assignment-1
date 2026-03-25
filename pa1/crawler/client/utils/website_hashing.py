import hashlib
import re
from bs4 import BeautifulSoup
import ppdeep


def hash_website(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    main_content = soup.find("article")
    to_hash = None
    if main_content:
        comments = main_content.find(id="comments_placeholder")
        if comments:
            # print("Comments Decomposed")
            comments.decompose()
        promo = main_content.find("div", attrs={"data-upscore-zone" : "microsite_promo"})
        if promo:
            # print("Promos Decomposed")
            promo.decompose()
        to_hash = main_content
    
    if to_hash == None:
        main_content = soup.find("main", class_="main")
        if main_content:
            to_hash = main_content



    to_hash = to_hash.get_text(separator=" ", strip=True)
    to_hash = re.sub(r"\s+", " ", to_hash)
    # text = text[:10000]
    #print(to_hash[:10000])
    h = ppdeep.hash(to_hash)
    
    # return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h