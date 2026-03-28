import hashlib
import re
from bs4 import BeautifulSoup
import ppdeep


def hash_website(html: str, url: str) -> str:
    soup = BeautifulSoup(html, features="html.parser")
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
    
    keywords = []
    if to_hash == None:
        # this is not an article then, so check for clues what kind of page it is
        
        # check the contents of the submenu
        submenu_div = soup.find("div", class_="submenu")
        if submenu_div:
            
            # extract all submenu titles
            submenu_items = submenu_div.find_all("a")
            for item in submenu_items:
                keywords.append(item.text)

            # extract the currently marked submenu tab
            active_submenu_item = submenu_div.find("a", class_="submenu__item-active")

            if active_submenu_item:
                keywords.append(active_submenu_item.text)

        pageContent = soup.find(id="pageContent")
        main_div = soup.find("main")

        ids = set()
        for el in pageContent.find_all(attrs={"id" : True}):
            id = el["id"]
            if "clip" in id or "div-gpt" in id or "Layer" in id:
                continue
            no_digits_id = ''.join([i for i in id if not i.isdigit()])
            if no_digits_id == "":
                continue
            ids.add(no_digits_id)
        
        keywords.extend(list(ids))

        if pageContent and not main_div:
            # if it doesnt have main tag, then its a smaller page, where we can take the titles
            uppercase_spans = pageContent.find_all("span", class_="uppercase")
            for span in uppercase_spans:
                keywords.append(span.text)


        if pageContent:
            h1s = pageContent.find_all("h1")
            # print(h1s)
            for h1 in h1s:
                keywords.append(h1.text)

        if main_div:
            keywords.append("has_main")




    if to_hash:
        to_hash = to_hash.get_text(separator=" ", strip=True)
        to_hash = re.sub(r"\s+", " ", to_hash)
    else:
        to_hash = " ".join(keywords)
    # print(to_hash)
    # text = text[:10000]
    #print(to_hash[:10000])
    h = ppdeep.hash(to_hash)
    
    # return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h