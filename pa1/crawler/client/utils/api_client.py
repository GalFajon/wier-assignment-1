from typing import Optional, Dict, Any, List
import requests
import threading
import os
from requests.auth import HTTPBasicAuth
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

class APIClient:
    def __init__(
        self,
        base_url: str,
        timeout: int = 10,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._local = threading.local()

        self._username = username if username is not None else os.environ.get("API_USER", "crawler")
        self._password = password if password is not None else os.environ.get("API_PASSWORD", "supersecret")

    def _debug_session_identity(self) -> dict:
        session = self._get_session()
        return {
            "thread_name": threading.current_thread().name,
            "thread_id": threading.get_ident(),
            "session_id": id(session)
        }

    def _get_session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
            self._local.session.auth = HTTPBasicAuth(self._username, self._password)
            self._local.session.verify = False
        return self._local.session

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # --- Sites ---
    def list_sites(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/sites/"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def get_site(self, site_id: int) -> Dict[str, Any]:
        r = self._get_session().get(self._url(f"/sites/{site_id}"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def create_site(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(self._url("/sites/"), json=payload, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def update_site(self, site_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().put(self._url(f"/sites/{site_id}"), json=payload, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def delete_site(self, site_id: int) -> None:
        r = self._get_session().delete(self._url(f"/sites/{site_id}"), timeout=self.timeout, verify=False)
        r.raise_for_status()

    def get_site_id_by_domain(self, domain: str) -> Optional[int]:
        r = self._get_session().get(
            self._url("/sites/by-domain"),
            params={"domain": domain},
            timeout=self.timeout, 
            verify=False
        )

        if r.status_code == 404:
            return None

        r.raise_for_status()
        return r.json().get("id")

    # --- Pages ---
    def list_pages(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/pages/"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def get_page(self, page_id: int) -> Dict[str, Any]:
        r = self._get_session().get(self._url(f"/pages/{page_id}"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()
    
    def get_page_id_by_url(self, url: str) -> Optional[int]:
        r = self._get_session().get(
            self._url("/pages/by-url"),
            params={"url": url},
            timeout=self.timeout, 
            verify=False
        )

        if r.status_code == 404:
            return None

        r.raise_for_status()
        return r.json().get("id")
    
    def get_page_id_by_hash(self, hash: str) ->  Optional[int]:
        r = self._get_session().get(
            self._url("/pages/by-hash"),
            params={"content_hash": hash},
            timeout=self.timeout,
            verify=False
        )

        if r.status_code == 404:
            return None

        r.raise_for_status()
        return r.json().get("id")
    
    def get_page_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        r = self._get_session().get(
            self._url("/pages/by-url"),
            params={"url": url},
            timeout=self.timeout, 
            verify=False
        )

        if r.status_code == 404:
            return None

        r.raise_for_status()
        return r.json()


    def create_page(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(
            self._url("/pages/"),
            json=payload,
            timeout=self.timeout, 
            verify=False
        )

        if r.status_code >= 400:
            # print("STATUS:", r.status_code)
            # print("RESPONSE:", r.text[:300]) # WIP REMOVE LATER
            try:
                error_body = r.json()
            except Exception:
                error_body = {"raw": r.text}

            raise requests.exceptions.HTTPError(
                f"APIClient - create_page - HTTP {r.status_code} Error | {error_body}",
                response=r
            )

        return r.json()
    
    def create_frontier_pages(self, payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        r = self._get_session().post(
            self._url("/pages/frontier/"),
            json=payload,
            timeout=self.timeout, 
            verify=False
        )

        if r.status_code >= 400:
            # print("STATUS:", r.status_code)
            # print("RESPONSE:", r.text[:300]) # WIP REMOVE LATER
            try:
                error_body = r.json()
            except Exception:
                error_body = {"raw": r.text}

            raise requests.exceptions.HTTPError(
                f"APIClient - create_frontier_pages - HTTP {r.status_code} Error | {error_body}",
                response=r
            )
        return r.json()
    
    def create_frontier_page(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(
            self._url("/pages/frontier_single/"),
            json=payload,
            timeout=self.timeout, 
            verify=False
        )

        if r.status_code >= 400:
            # print("STATUS:", r.status_code)
            # print("RESPONSE:", r.text[:300]) # WIP REMOVE LATER
            try:
                error_body = r.json()
            except Exception:
                error_body = {"raw": r.text}
            raise requests.exceptions.HTTPError(
                f"APIClient - create_frontier_page - HTTP {r.status_code} Error | {error_body}",
                response=r
            )
        return r.json()
    
    def update_frontier_pages(self, payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        r = self._get_session().put(
            self._url("/pages/frontier/"),
            json=payload,
            timeout=self.timeout, 
            verify=False
        )

        if r.status_code >= 400:
            # print("STATUS:", r.status_code)
            # print("RESPONSE:", r.text[:300]) # WIP REMOVE LATER
            try:
                error_body = r.json()
            except Exception:
                error_body = {"raw": r.text}

            raise requests.exceptions.HTTPError(
                f"APIClient - update_frontier_pages - HTTP {r.status_code} Error | {error_body}",
                response=r
            )

        return r.json()
    
    def update_frontier_page(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().put(
            self._url("/pages/frontier_single/"),
            json=payload,
            timeout=self.timeout,
            verify=False
        )

        if r.status_code >= 400:
            # print("STATUS:", r.status_code)
            # print("RESPONSE:", r.text[:300]) # WIP REMOVE LATER
            try:
                error_body = r.json()
            except Exception:
                error_body = {"raw": r.text}

            raise requests.exceptions.HTTPError(
                f"APIClient - update_frontier_page - HTTP {r.status_code} Error | {error_body}",
                response=r
            )

        return r.json()
    
    

    def update_page(self, page_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().put(self._url(f"/pages/{page_id}"), json=payload, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def delete_page(self, page_id: int) -> None:
        r = self._get_session().delete(self._url(f"/pages/{page_id}"), timeout=self.timeout, verify=False)
        r.raise_for_status()

    # --- PageData ---
    def list_page_data(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/page_data/"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def get_page_data(self, pd_id: int) -> Dict[str, Any]:
        r = self._get_session().get(self._url(f"/page_data/{pd_id}"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def create_page_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(self._url("/page_data/"), json=payload, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def create_page_data_file(
        self,
        page_id: Optional[int],
        data_type_code: Optional[str],
        filename: str,
        file_bytes: bytes,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        files = {"file": (filename, file_bytes, content_type or "application/octet-stream")}
        data: Dict[str, str] = {}

        if page_id is not None:
            data["page_id"] = str(page_id)
        if data_type_code is not None:
            data["data_type_code"] = data_type_code

        r = self._get_session().post(self._url("/page_data/"), files=files, data=data, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def update_page_data(self, pd_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().put(self._url(f"/page_data/{pd_id}"), json=payload, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def update_page_data_file(
        self,
        pd_id: int,
        filename: str,
        file_bytes: bytes,
        content_type: Optional[str] = None,
        page_id: Optional[int] = None,
        data_type_code: Optional[str] = None
    ) -> Dict[str, Any]:
        files = {"file": (filename, file_bytes, content_type or "application/octet-stream")}
        data: Dict[str, str] = {}

        if page_id is not None:
            data["page_id"] = str(page_id)
        if data_type_code is not None:
            data["data_type_code"] = data_type_code

        r = self._get_session().put(self._url(f"/page_data/{pd_id}"), files=files, data=data, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def delete_page_data(self, pd_id: int) -> None:
        r = self._get_session().delete(self._url(f"/page_data/{pd_id}"), timeout=self.timeout, verify=False)
        r.raise_for_status()

    # --- Links ---
    def list_links(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/links/"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def list_links_with_urls(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/links/with-urls/"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def create_link(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(self._url("/links/"), json=payload, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def delete_link(self, from_page: int, to_page: int) -> None:
        r = self._get_session().delete(self._url(f"/links/{from_page}/{to_page}"), timeout=self.timeout, verify=False)
        r.raise_for_status()

    # --- Images ---
    def create_image_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(self._url("/images/"), json=payload, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def create_image_file(
        self,
        page_id: int,
        filename: str,
        file_bytes: bytes,
        content_type: Optional[str] = None,
        accessed_time: Optional[str] = None
    ) -> Dict[str, Any]:
        files = {"file": (filename, file_bytes, content_type or "application/octet-stream")}
        data = {"page_id": str(page_id)}

        if accessed_time is not None:
            data["accessed_time"] = accessed_time

        r = self._get_session().post(self._url("/images/"), files=files, data=data, timeout=self.timeout, verify=False)
        r.raise_for_status()

        if r.content:
            try:
                return r.json()
            except Exception:
                return {"status": "ok"}
        return {"status": "ok"}

    def list_images(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/images/"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def get_image(self, image_id: int) -> Dict[str, Any]:
        r = self._get_session().get(self._url(f"/images/{image_id}"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def update_image_json(self, image_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().put(self._url(f"/images/{image_id}"), json=payload, timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()

    def delete_image(self, image_id: int) -> None:
        r = self._get_session().delete(self._url(f"/images/{image_id}"), timeout=self.timeout, verify=False)
        r.raise_for_status()

    def _do_not_use_health(self) -> Dict[str, Any]:
        r = self._get_session().get(self._url("/health"), timeout=self.timeout, verify=False)
        r.raise_for_status()
        return r.json()