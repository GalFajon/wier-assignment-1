import logging
import sys
import time
from typing import List
import requests
import threading
from queue import PriorityQueue

from urllib.parse import urlsplit
from robotexclusionrulesparser import RobotExclusionRulesParser
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait

from utils.priority_scoring import priority_score  # type: ignore
from utils.url_cleaning import normalize_url # type: ignore
from utils.website_parsing import parse_website_content # type: ignore

class WebCrawler24Ur:

    def __init__(
        self,
        seed_urls: List[str],
        crawler_id: str = "fri-ieps-24Ur-crawler",
        page_timeout_seconds: int = 15,
        max_pages: int = 10,
        worker_count: int = 4,
        web_driver_location: str = "/usr/local/bin/geckodriver",
        default_crawl_delay: float = 1.0,
        logging_level: str = 'DEBUG',
        logging_file: str = './crawler.log',
        log_to_stdout: bool = True
    ) -> None:
        self._seed_urls = seed_urls
        self._max_pages = max_pages
        self._page_timeout_seconds = page_timeout_seconds
        self._crawler_id = crawler_id
        self._worker_count = worker_count
        self._web_driver_location = web_driver_location

        # setup logging
        self._logger = logging.getLogger(crawler_id)
        numeric_level = getattr(logging, logging_level.upper(), logging.INFO)
        self._logger.setLevel(numeric_level)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"
        )

        if not self._logger.handlers:
            file_handler = logging.FileHandler(logging_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            self._logger.addHandler(file_handler)
            if log_to_stdout:
                console = logging.StreamHandler(sys.stdout)
                console.setFormatter(formatter)
                console.setLevel(numeric_level)
                self._logger.addHandler(console)

        # domain prep
        self._domains = []
        for seed in seed_urls:
            seed_url_parts = urlsplit(seed)
            self._domains.append(seed_url_parts.scheme + "://" + seed_url_parts.netloc)

        # shared structures and locks
        self._shared_visited_urls = set()
        self._lock_visited_urls = threading.Lock()

        self._shared_crawling_front = PriorityQueue()

        self._shared_downloaded_page_count = 0
        self._lock_downloaded_page_count = threading.Lock()

        self._shared_robots_info = {}
        self._shared_last_access = {}
        self._lock_website_access_info = threading.Lock()

        # robots.txt info
        for domain in self._domains:
            robots_url = domain.rstrip("/") + "/robots.txt"
            try:
                r = requests.get(robots_url, timeout=5)
                rp = RobotExclusionRulesParser()
                rp.parse(r.text)

                delay = rp.get_crawl_delay(self._crawler_id)

                if delay is None:
                    delay = rp.get_crawl_delay("*")

                if delay is None:
                    delay = default_crawl_delay

                self._shared_robots_info[domain] = {
                    "info": rp,
                    "delay": delay
                }
            except Exception:
                self._shared_robots_info[domain] = {
                    "info": None,
                    "delay": default_crawl_delay
                }
            self._shared_last_access[domain] = time.time()

        self._logger.info(f"INITIALIZED CRAWLER")
        delay_info = {d: v["delay"] for d, v in self._shared_robots_info.items()}
        self._logger.debug(f"Crawl delays: {delay_info}" )


        # initialize queue
        for seed in seed_urls:
            self._shared_crawling_front.put((0, seed))






    def _create_driver(self, worker_id):
        firefox_options = FirefoxOptions()
        firefox_options.add_argument("--headless")
        firefox_options.set_preference(
            "general.useragent.override",
            f'{self._crawler_id}-{worker_id}'
        )
        service = Service(executable_path=self._web_driver_location)

        return webdriver.Firefox(
            service=service,
            options=firefox_options
        )




    def _fetch_page(self, worker_driver, url):
        try:
            worker_driver.get(url)
            WebDriverWait(worker_driver, self._page_timeout_seconds).until(
                lambda d: d.execute_script(
                    "return document.readyState"
                ) == "complete"
            )
            return worker_driver.page_source
        except Exception:
            return None


    def _valid_url(self, url):
        with self._lock_visited_urls:
            if url in self._shared_visited_urls:
                return False

        url_parts = urlsplit(url)
        host = url_parts.netloc
        domain = url_parts.scheme + "://" + host

        allowed_hosts = {urlsplit(d).netloc for d in self._domains}
        if host not in allowed_hosts:
            return False

        domain_info = self._shared_robots_info.get(domain)
        rp = domain_info['info']
        if rp is not None and not rp.is_allowed(self._crawler_id, url):
            self._logger.info('URL:', url, "is NOT allowed by robots.txt")
            return False

        return True




    def _respect_crawl_delay(self, domain):
        with self._lock_website_access_info:
            info = self._shared_robots_info.get(domain)
            delay = info["delay"]

            now = time.time()
            next_allowed = self._shared_last_access[domain]

            if now < next_allowed:
                wait = next_allowed - now
            else:
                wait = 0

            self._shared_last_access[domain] = max(now, next_allowed) + delay

        if wait > 0:
            time.sleep(wait)






    def _deploy_crawl_worker(self, worker_id):
        worker_web_driver = self._create_driver(worker_id)
        worker_name = threading.current_thread().name

        while True:
            try:
                priority, url = self._shared_crawling_front.get(timeout=3)
            except:
                break
            
            # validate url
            if not self._valid_url(url):
                self._shared_crawling_front.task_done()
                continue
            
            # end if reached page count max
            with self._lock_downloaded_page_count:
                if self._shared_downloaded_page_count >= self._max_pages:
                    self._shared_crawling_front.task_done()
                    break
                self._shared_downloaded_page_count += 1

            # process url
            url_parts = urlsplit(url)
            domain = url_parts.scheme + "://" + url_parts.netloc

            self._respect_crawl_delay(domain)
            html = self._fetch_page(worker_web_driver, url)

            # invalid response - no body
            if not html:
                self._shared_crawling_front.task_done()
                continue
            
            # add to visited set
            with self._lock_visited_urls:
                self._shared_visited_urls.add(url)


            website_data = parse_website_content(html)
            website_urls = website_data['urls']
            
            self._logger.info(f"[{worker_name}] Crawled:   {url}")
            self._logger.info(f"  - Found {len(website_urls)} links")

            for url_data in website_urls:
                link, tag = url_data[0], url_data[1] # dummy simple unclean dat
                link_norm = normalize_url(link)

                with self._lock_visited_urls:
                    if link_norm in self._shared_visited_urls:
                        continue

                priority = priority_score(html, url_data)
                self._shared_crawling_front.put((priority, link_norm))

            self._shared_crawling_front.task_done()

        worker_web_driver.quit()



    def crawl(self):
        self._logger.info(f"Beginning crawl")

        worker_threads = []
        for worker_id in range(self._worker_count):
            t = threading.Thread(
                target=self._deploy_crawl_worker,
                args=(worker_id,),
                name=f"Worker-{worker_id}"
            )

            t.start()
            worker_threads.append(t)

        for t in worker_threads:
            t.join()





if __name__ == "__main__":

    seed = "https://www.24ur.com/"

    crawler = WebCrawler24Ur(
        seed_urls=[seed],
        max_pages=10,
        worker_count=4,
        log_to_stdout=True,
        logging_file='./crawler.log',
        logging_level='DEBUG'
    )

    crawler.crawl()