import yaml
from utils.api_client import APIClient
from tqdm import tqdm

if __name__ == '__main__':

    db_client = APIClient(base_url='https://server:5000', timeout=200)

    pages = db_client.list_pages()

    data = []

    for page in tqdm(pages, desc="Fetching pages", total=len(pages)):
        if page['page_type_code'] != 'HTML':
            continue

        entry = {
            "id": page["id"],
            "url": page["url"],
            "html_content": page["html_content"]
        }
        data.append(entry)

    with open("/client/output/pages.yaml", "w", encoding="utf-8") as f:
        for page in tqdm(data, desc="Saving YAML", total=len(data)):
            yaml.dump(
                [page],
                f,
                allow_unicode=True,
                sort_keys=False
            )