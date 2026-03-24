from utils.url_cleaning import canonicalize_url
from utils.page_data_objects import PageDbSaveObject, LinkData, ImageData, PageData
from utils.website_parsing import get_page_database_save_object
from utils.api_client import APIClient



if __name__ == '__main__':


    db_client = APIClient(base_url='http://server:5000')

    #clear sites
    print('Clearing sites')
    for site in db_client.list_sites():
        id = site['id']
        response = db_client.delete_site(id)
        print(f"cleared {id}")

    #clear pages
    print('Clearing pages')
    for site in db_client.list_pages():
        id = site['id']
        response = db_client.delete_page(id)
        print(f"cleared {id}")


    
    # response = db_client.list_sites()
    # print(f"Resp: {response}")

    # response = db_client.list_pages()
    # print(f"Resp: {response}")

    # domain = "https://www.24ur.com/"
    # response = db_client.get_site_id_by_domain(domain)
    # print(f"Resp: {response}")

    # response = db_client.create_site({
    #     'domain' : domain,
    #     'robots_content' : '',
    #     'sitemap_content' : ''
    # })
    # print(f"Resp: {response}")

    # response = db_client.get_site_id_by_domain(domain)
    # print(f"Resp: {response}")