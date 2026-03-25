import yaml
from utils.api_client import APIClient

if __name__ == '__main__':

    db_client = APIClient(base_url='http://server:5000')

    pages = db_client.list_pages()

    data = []

    for page in pages:
        entry = {
            "id": page["id"],
            "url": page["url"],
            "html_content": page["html_content"]
        }
        data.append(entry)

    with open("/client/output/pages.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)