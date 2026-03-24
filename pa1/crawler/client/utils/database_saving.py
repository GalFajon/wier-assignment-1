
from utils.url_cleaning import canonicalize_url
from utils.page_data_objects import PageDbSaveObject, LinkData, ImageData, PageData
from utils.website_parsing import get_page_database_save_object
from utils.api_client import APIClient
from requests.exceptions import HTTPError


def get_site_id_or_create_site(logger, domain, db_api: APIClient):
    try:
        site_id = db_api.get_site_id_by_domain(domain)

        if site_id is not None:
            return site_id

        logger.debug(f"Site not found, creating: {domain}")

        response = db_api.create_site({
            "domain": domain,
            "robots_content": "",
            "sitemap_content": ""
        })

        return response["id"]

    except Exception as e:
        logger.error(f"Failed to get/create site for {domain}: {e}")
        raise


def save_page_or_update(logger, page_payload, db_api: APIClient):
    try:
        page_json = db_api.create_page(page_payload)
        logger.debug(f"Created page: {page_json['id']} {page_json['url']}")
        return page_json

    except HTTPError as e:
        if e.response is not None and e.response.status_code == 400:

            logger.debug("Page URL already exists, resolving URL duplicate")

            try:
                page_id = db_api.get_page_id_by_url(page_payload["url"])

                if page_id == None:
                    logger.error("Duplicate detected but page not found")
                    return None

                logger.debug(f"Updating existing page id={page_id}")
                return db_api.update_page(page_id, page_payload)

            except Exception as inner_e:
                logger.error(f"Failed to update existing page: {inner_e}")
                return None

        logger.error(f"Unexpected HTTP error: {e}")
        raise

def save_link(logger, from_page_id: int, to_page_id: int, db_api: APIClient) -> bool:
    if from_page_id == to_page_id:
        logger.debug(f"Skipping self-link ({from_page_id} -> {to_page_id})")
        return False

    link_payload = {
        "from_page": from_page_id,
        "to_page": to_page_id
    }

    try:
        resp = db_api.create_link(link_payload)
        logger.debug(f"Created link: {resp}")
        return True

    except HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            logger.debug(f"Invalid Link Pair or Already Exists: {from_page_id} -> {to_page_id}")
            return True

        logger.error(f"Unexpected error creating link: {e}")
        raise

def save_page_to_db(logger, url, html, from_page_id, db_api: APIClient):
    url_norm = canonicalize_url(url)

    logger.info(f"Saving {url_norm} to DB ({url})")

    database_save_object: PageDbSaveObject = get_page_database_save_object(logger, url, html)

    #logger.debug(f"Database save object: {database_save_object}")

    if database_save_object is None:
        logger.warning("Database save object IS NOT VALID")
        return -1

    site_id = get_site_id_or_create_site(logger, database_save_object.site_domain, db_api)

    page_payload = {
        "site_id": site_id,
        "page_type_code": database_save_object.page_type_code,
        "url": database_save_object.url,
        "html_content": database_save_object.html_content,
        "http_status_code": database_save_object.http_status_code,
        'accessed_time': database_save_object.accessed_time.isoformat()
    }

    debug_payload = dict(page_payload)
    debug_payload["html_content"] = "..."
    logger.debug(f"Saving {debug_payload}, hash={database_save_object.content_hash}")

    page_json_data = save_page_or_update(logger, page_payload, db_api)
    if page_json_data == None:
        logger.error(f"ERROR DURING PAGE DB SAVE")
        return -1
    
    page_id = page_json_data['id']

    # image saving
    logger.debug(f"Saving {len(database_save_object.images)} images.")
    for image in database_save_object.images:
        image_payload = {
            'page_id' : page_id,
            'filename' : image.filename,
            'content_type' : image.content_type,
            'data' : None,
            'accessed_time' : image.accessed_time.isoformat()
        }

        resp = db_api.create_image_json(image_payload)
        #logger.debug(f"Saving image response {resp}")

    # page data saving
    logger.debug(f"Saving {len(database_save_object.page_data)} page data entries.")
    for page_data_entry in database_save_object.page_data:
        page_data_entry_payload = {
            'page_id' : page_id,
            'data_type_code' : page_data_entry.data_type_code,
            'data' : None,
        }

        resp = db_api.create_page_data(page_data_entry_payload)
        #logger.debug(f"Saving page data response {resp}")

    # link saving
    logger.debug(f"Saving Link.")
    save_link(logger, from_page_id, page_id, db_api)

    return page_id

