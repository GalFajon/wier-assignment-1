from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime

def _shorten(s: str, max_len: int = 120) -> str:
    if s is None:
        return "None"
    return (s[:max_len] + "...") if len(s) > max_len else s

@dataclass
class PageData:
    data_type_code: str
    data: bytes

    def __str__(self):
        return f"PageData(type={self.data_type_code}, bytes={len(self.data)})"

@dataclass
class ImageData:
    filename: str
    content_type: str
    data: bytes
    accessed_time: datetime

    def __str__(self):
        return (
            f"ImageData(filename={self.filename}, "
            f"type={self.content_type}, "
            f"bytes={len(self.data)}, "
            f"time={self.accessed_time})"
        )

@dataclass
class LinkData:
    to_url: str

    def __str__(self):
        return f"Link(to={_shorten(self.to_url, 100)})"

@dataclass
class PageDbSaveObject:
    #req
    url: str
    site_domain: str
    page_type_code: str   # HTML, BINARY, DUPLICATE, FRONTIER

    #optional
    html_content: Optional[str] = None
    http_status_code: Optional[int] = None
    accessed_time: Optional[datetime] = None

    # relational data
    page_data: List[PageData] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    links: List[LinkData] = field(default_factory=list)

    # for deduplication
    content_hash: Optional[str] = None
    duplicate_of_url: Optional[str] = None
    priority: Optional[float] = None

    def is_html(self) -> bool:
        return self.page_type_code == "HTML"

    def is_binary(self) -> bool:
        return self.page_type_code == "BINARY"

    def is_duplicate(self) -> bool:
        return self.page_type_code == "DUPLICATE"

    def add_link(self, to_url: str):
        self.links.append(LinkData(to_url=to_url))

    def add_image(self, filename: str, content_type: str, data: bytes, accessed_time: datetime):
        self.images.append(
            ImageData(
                filename=filename,
                content_type=content_type,
                data=data,
                accessed_time=accessed_time
            )
        )

    def add_page_data(self, data_type_code: str, data: bytes):
        self.page_data.append(
            PageData(
                data_type_code=data_type_code,
                data=data
            )
        )

    def __str__(self):
        lines = []

        lines.append("PageDbSaveObject(")
        lines.append(f"  url={_shorten(self.url)}")
        lines.append(f"  site_domain={self.site_domain}")
        lines.append(f"  type={self.page_type_code}")
        lines.append(f"  status={self.http_status_code}")
        lines.append(f"  accessed={self.accessed_time}")

        lines.append(f"  content_hash={self.content_hash}")
        lines.append(f"  duplicate_of={self.duplicate_of_url}")

        if self.html_content:
            lines.append(f"  html_length={len(self.html_content)}")
        else:
            lines.append("  html_length=None")

        lines.append(f"  links={len(self.links)}")
        lines.append(f"  images={len(self.images)}")
        lines.append(f"  page_data={len(self.page_data)}")

        if self.links:
            lines.append("  link_preview=[")
            for l in self.links[:5]:
                lines.append(f"    {l}")
            if len(self.links) > 5:
                lines.append("    ...")
            lines.append("  ]")

        if self.images:
            lines.append("  image_preview=[")
            for img in self.images[:3]:
                lines.append(f"    {img}")
            if len(self.images) > 3:
                lines.append("    ...")
            lines.append("  ]")

        if self.page_data:
            lines.append("  page_data_preview=[")
            for d in self.page_data[:3]:
                lines.append(f"    {d}")
            if len(self.page_data) > 3:
                lines.append("    ...")
            lines.append("  ]")

        lines.append(")")

        return "\n".join(lines)