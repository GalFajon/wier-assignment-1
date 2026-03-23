
from utils.url_cleaning import canonicalize_url
from utils.page_data_objects import PageDbSaveObject, LinkData, ImageData, PageData
from utils.website_parsing import get_page_database_save_object
from utils.api_client import APIClient

def save_page_to_db(logger, url, html, db_api : APIClient):
    url_norm = canonicalize_url(url)

    logger.info(f"Saving {url_norm} to DB ({url})")
    database_save_object : PageDbSaveObject = get_page_database_save_object(logger, url, html)
    #logger.debug(f"Check Session: {db_api._debug_session_identity()}")

    #WIP - turn DSobject parsing into APIClient calls

    return True