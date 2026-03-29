import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm
import csv
import sys
from bs4 import BeautifulSoup

BLACKLIST_PATTERNS = [
    "hitro in učinkovito iskanje željene vsebine",
    "nabor vsebine glede na iskano temo",
    "upravljavec vaših podatkov je pro plus",
    "uporabljamo piškotke",
    "partnerjev uporabljamo",
    "pravilniku o zasebnosti",
    "politiki piškotkov",
    "obdelavo podatkov",
    "oglaševanje in vsebina",
]

def is_bad_text(text):
    text_norm = text.lower().strip()
    return any(p in text_norm for p in BLACKLIST_PATTERNS)


def extract_text(html):
    soup = BeautifulSoup(html, "lxml")

    # Remove obvious junk
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Remove cookie/consent containers aggressively
    for div in soup.find_all(["div", "section"]):
        text = div.get_text(" ", strip=True).lower()
        if any(p in text for p in BLACKLIST_PATTERNS):
            div.decompose()

    title = soup.title.get_text(strip=True) if soup.title else None

    # ---- SUMMARY ----
    summary = None
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        summary = meta["content"].strip()

    if not summary:
        p = soup.find("p")
        if p:
            summary = p.get_text(strip=True)

    if summary and is_bad_text(summary):
        summary = None

    # ---- BODY ----
    article = soup.find("article")

    if not article:
        candidates = soup.find_all("div")

        # Only consider divs with meaningful paragraph content
        candidates = [
            div for div in candidates
            if len(div.find_all("p")) >= 3
        ]

        article = max(
            candidates,
            key=lambda tag: len(tag.get_text(strip=True)),
            default=None
        )

    body = ""

    if article:
        paragraphs = []
        for p in article.find_all("p"):
            text = p.get_text(strip=True)

            # Filter junk paragraphs
            if not text:
                continue
            if is_bad_text(text):
                continue
            if len(text) < 40:  # removes short UI fragments
                continue

            paragraphs.append(text)

        body = "\n".join(paragraphs)

    # Final safeguard: discard garbage pages
    if is_bad_text(body[:500]):
        body = ""

    return {
        "title": title,
        "summary": summary,
        "body": body
    }






if __name__ == "__main__":
    INPUT = "page_table.csv"
    OUTPUT = "cleaned_pages.yaml"


    # IDS_FILE = "cluster_19_ids.txt"
    # with open(IDS_FILE, "r", encoding="utf-8") as f:
    #     target_ids = set(line.strip() for line in f if line.strip())

    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int //= 10

    cleaned = []

    with open(INPUT, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for page in tqdm(reader, desc="Cleaning pages"):
            if not page.get("html_content"):
                continue

            # if str(page.get("id")) not in target_ids:
            #     continue


            extracted = extract_text(page["html_content"])

            cleaned.append({
                "id": page["id"],
                "url": page["url"],
                **extracted
            })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        yaml.dump(cleaned, f, allow_unicode=True, sort_keys=False)