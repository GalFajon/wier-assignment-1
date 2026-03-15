from urllib.parse import urlsplit, urljoin
import urllib.robotparser
from robotexclusionrulesparser import RobotExclusionRulesParser
import requests
from bs4 import BeautifulSoup
import heapq
import numpy as np
from urllib.parse import urlsplit
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List


class WebCrawler24Ur:
    def __init__(
        self,
        seed_urls: List[str],
        crawler_id: str = "fri-ieps-24Ur-crawler",
        page_timeout_seconds: int = 15,
        max_pages: int = 10,
        web_driver_location: str = "/usr/local/bin/geckodriver"
    ) -> None:
        
        # basic params
        self._seed_urls = seed_urls
        self._max_pages = max_pages
        self._page_timeout_seconds = page_timeout_seconds
        self._crawler_id = crawler_id

        # domain prep
        self._domains = []
        for seed in seed_urls:
            seed_url_parts = urlsplit(seed)
            self._domains.append(seed_url_parts.scheme + "://" + seed_url_parts.netloc)

        # selenium setup
        firefox_options = FirefoxOptions()
        firefox_options.add_argument("--headless")
        firefox_options.set_preference(
            "general.useragent.override",
            crawler_id
        )
        service = Service(executable_path=web_driver_location)
        self._web_driver = webdriver.Firefox(
            service=service,
            options=firefox_options
        )

        # manage robots.txt info
        self._robots_info = {}
        self._crawl_delay = {}
        self._last_access = {}

        for domain in self._domains:
            robots_url = domain.rstrip("/") + "/robots.txt"
            try:
                r = requests.get(robots_url, timeout=5)
                rp = RobotExclusionRulesParser()
                rp.parse(r.text)
                self._robots_info[domain] = rp

                delay = rp.get_crawl_delay(self._crawler_id)
                if delay is None:
                    delay = rp.get_crawl_delay("*")

                if delay is None:
                    delay = 0.0

                self._crawl_delay[domain] = delay

            except Exception:
                self._robots_info[domain] = None
                self._crawl_delay[domain] = 0.0

            self._last_access[domain] = 0.0

        print(self._crawl_delay)


        # front management
        self.visited = set()
        self.queue = []
        for seed in seed_urls:
            heapq.heappush(self.queue, (0, seed))




    def _fetch_page(self, url):
        try:
            self._web_driver.get(url)
            WebDriverWait(self._web_driver, self._page_timeout_seconds).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            return self._web_driver.page_source
        except Exception:
            return None


    def _parse_page(self, html, base_url):
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("http"):
                links.append((href, a_tag))
            elif href.startswith("//"):
                links.append(("https:" + href, a_tag))
            elif href.startswith("/"):
                links.append((base_url + href, a_tag))
        return links

    def _valid_url(self, url):

        # already visited
        if url in self.visited:
            return False

        url_parts = urlsplit(url)
        host = url_parts.netloc
        domain = url_parts.scheme + "://" + host

        allowed_hosts = {urlsplit(d).netloc for d in self._domains}

        # outside allowed domains
        if host not in allowed_hosts:
            return False

        rp = self._robots_info.get(domain)

        # not allowed by robots.txt
        if rp is not None and not rp.is_allowed(self._crawler_id, url):
            print("Not allowed by robots:", self._crawler_id, url)
            return False

        return True



    def _priority(self, page_entity):
        return 1


    def crawl(self):

        pages_crawled = 0
        while self.queue and pages_crawled < self._max_pages:
            priority, url = heapq.heappop(self.queue)

            if not self._valid_url(url):
                continue

            html = self._fetch_page(url)
            if not html:
                continue

            self.visited.add(url)

            url_parts = urlsplit(url)
            base_url = url_parts.scheme + "://" + url_parts.netloc
            links = self._parse_page(html, base_url)
            print("  - Found", len(links), "links")
            for link, link_tag in links:
                if link not in self.visited:
                    priority = self._priority(html)
                    heapq.heappush(self.queue, (priority, link))

            pages_crawled += 1
    
    def close(self):
        if self._web_driver:
            self._web_driver.quit()


if __name__ == "__main__":
    seed = "https://en.wikipedia.org/wiki/Albert_Einstein"  # Replace with an actual URL
    crawler = WebCrawler24Ur(seed_urls=[seed], max_pages=5)
    crawler.crawl()
    crawler.close()