
from utils.url_cleaning import canonicalize_url
from utils.page_data_objects import PageDbSaveObject, LinkData, ImageData, PageData
from utils.website_parsing import get_page_database_save_object
from utils.api_client import APIClient

def save_page_to_db(logger, url, html, db_api : APIClient):
    url_norm = canonicalize_url(url)

    logger.info(f"Saving {url_norm} to DB ({url})")
    database_save_object : PageDbSaveObject = get_page_database_save_object(logger, url, html)
    for i in database_save_object.images:
        logger.debug(f"Database save object: {i}")
    #logger.debug(f"Check Session: {db_api._debug_session_identity()}")

    #WIP - turn DSobject parsing into APIClient calls
    response = db_api.create_page({
        "site_id": 0, # TODO: update with actual site id
        "page_type_code": database_save_object.page_type_code,
        "url": database_save_object.url,
        "html_content": database_save_object.html_content,
        "http_status_code": database_save_object.http_status_code,
        "accessed_time": str(database_save_object.accessed_time)
    })

    return True