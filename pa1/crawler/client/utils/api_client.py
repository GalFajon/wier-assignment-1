from typing import Optional, Dict, Any, List
import requests
import threading


class APIClient:
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._local = threading.local()

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
        return self._local.session

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # --- Sites ---
    def list_sites(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/sites/"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_site(self, site_id: int) -> Dict[str, Any]:
        r = self._get_session().get(self._url(f"/sites/{site_id}"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def create_site(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(self._url("/sites/"), json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def update_site(self, site_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().put(self._url(f"/sites/{site_id}"), json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def delete_site(self, site_id: int) -> None:
        r = self._get_session().delete(self._url(f"/sites/{site_id}"), timeout=self.timeout)
        r.raise_for_status()

    # --- Pages ---
    def list_pages(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/pages/"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_page(self, page_id: int) -> Dict[str, Any]:
        r = self._get_session().get(self._url(f"/pages/{page_id}"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def create_page(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(self._url("/pages/"), json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def update_page(self, page_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().put(self._url(f"/pages/{page_id}"), json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def delete_page(self, page_id: int) -> None:
        r = self._get_session().delete(self._url(f"/pages/{page_id}"), timeout=self.timeout)
        r.raise_for_status()

    # --- PageData ---
    def list_page_data(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/page_data/"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_page_data(self, pd_id: int) -> Dict[str, Any]:
        r = self._get_session().get(self._url(f"/page_data/{pd_id}"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def create_page_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(self._url("/page_data/"), json=payload, timeout=self.timeout)
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

        r = self._get_session().post(self._url("/page_data/"), files=files, data=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def update_page_data(self, pd_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().put(self._url(f"/page_data/{pd_id}"), json=payload, timeout=self.timeout)
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

        r = self._get_session().put(self._url(f"/page_data/{pd_id}"), files=files, data=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def delete_page_data(self, pd_id: int) -> None:
        r = self._get_session().delete(self._url(f"/page_data/{pd_id}"), timeout=self.timeout)
        r.raise_for_status()

    # --- Links ---
    def list_links(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/links/"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def create_link(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(self._url("/links/"), json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def delete_link(self, from_page: int, to_page: int) -> None:
        r = self._get_session().delete(self._url(f"/links/{from_page}/{to_page}"), timeout=self.timeout)
        r.raise_for_status()

    # --- Images ---
    def create_image_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().post(self._url("/images/"), json=payload, timeout=self.timeout)
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

        r = self._get_session().post(self._url("/images/"), files=files, data=data, timeout=self.timeout)
        r.raise_for_status()

        if r.content:
            try:
                return r.json()
            except Exception:
                return {"status": "ok"}
        return {"status": "ok"}

    def list_images(self) -> List[Dict[str, Any]]:
        r = self._get_session().get(self._url("/images/"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_image(self, image_id: int) -> Dict[str, Any]:
        r = self._get_session().get(self._url(f"/images/{image_id}"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def update_image_json(self, image_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = self._get_session().put(self._url(f"/images/{image_id}"), json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def delete_image(self, image_id: int) -> None:
        r = self._get_session().delete(self._url(f"/images/{image_id}"), timeout=self.timeout)
        r.raise_for_status()

    # --- Health ---
    def health(self) -> Dict[str, Any]:
        r = self._get_session().get(self._url("/health"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()