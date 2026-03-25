import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

def extract_text(html):
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = None
    if soup.title:
        title = soup.title.get_text(strip=True)

    summary = None

    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        summary = meta["content"].strip()

    if not summary:
        p = soup.find("p")
        if p:
            summary = p.get_text(strip=True)

    article = soup.find("article")

    if not article:
        candidates = soup.find_all("div")
        article = max(
            candidates,
            key=lambda tag: len(tag.get_text(strip=True)),
            default=soup
        )

    paragraphs = article.find_all("p")
    body = "\n".join(p.get_text(strip=True) for p in paragraphs)

    return {
        "title": title,
        "summary": summary,
        "body": body
    }


if __name__ == "__main__":
    INPUT = "pages.yaml"
    OUTPUT = "cleaned_pages.yaml"

    with open(INPUT, "r", encoding="utf-8") as f:
        pages = yaml.safe_load(f)

    cleaned = []

    for page in tqdm(pages, desc="Cleaning pages", total=len(pages)):
        extracted = extract_text(page["html_content"])

        cleaned.append({
            "id": page["id"],
            "url": page["url"],
            **extracted
        })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        yaml.dump(cleaned, f, allow_unicode=True, sort_keys=False)