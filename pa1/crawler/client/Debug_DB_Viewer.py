from utils.url_cleaning import canonicalize_url
from utils.page_data_objects import PageDbSaveObject, LinkData, ImageData, PageData
from utils.website_parsing import get_page_database_save_object
from utils.api_client import APIClient



if __name__ == '__main__':


    db_client = APIClient(base_url='http://server:5000')

    #clear page data
    print('Clearing links')
    for link in db_client.list_links():
        #print(link)
        fp = link['from_page']
        tp = link['to_page']
        response = db_client.delete_link(fp, tp)
        #print(f"cleared {response}")

    #clear images
    print('Clearing images')
    for image in db_client.list_images():
        id = image['id']
        response = db_client.delete_image(id)
        #print(f"cleared {id}")

    #clear pages
    print('Clearing pages')
    for page in db_client.list_pages():
        id = page['id']
        response = db_client.delete_page(id)
        #print(f"cleared {id}")

    #clear sites
    print('Clearing sites')
    for site in db_client.list_sites():
        id = site['id']
        response = db_client.delete_site(id)
        #print(f"cleared {id}")

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