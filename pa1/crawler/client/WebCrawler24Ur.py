import logging
import sys
import time
from typing import List
from bs4 import BeautifulSoup
import requests
import threading
from queue import PriorityQueue
import classla

from urllib.parse import urlsplit
from robotexclusionrulesparser import RobotExclusionRulesParser
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.support import expected_conditions as EC

from utils.priority_scoring import priority_score_BOW, priority_score_BERT, embed_BERT  # type: ignore
from utils.url_cleaning import normalize_url # type: ignore
from utils.website_parsing import parse_website_content # type: ignore
from utils.database_saving import save_page_to_db # type: ignore
from utils.api_client import APIClient

class WebCrawler24Ur:

    def __init__(
        self,
        seed_urls: List[str],
        crawler_id: str = "fri-ieps-24Ur-crawler",
        page_timeout_seconds: int = 15,
        max_pages: int = 10,
        worker_count: int = 4,
        scoring_method: str = 'BERT',
        database_base_url: str = 'http://server:5000',
        web_driver_location: str = "/usr/local/bin/geckodriver",
        default_crawl_delay: float = 1.0,
        logging_level: str = 'DEBUG',
        logging_file: str = './crawler.log',
        log_to_stdout: bool = True,
        query: str = ""
    ) -> None:
        self._seed_urls = seed_urls
        self._max_pages = max_pages
        self._page_timeout_seconds = page_timeout_seconds
        self._crawler_id = crawler_id
        self._worker_count = worker_count
        self._web_driver_location = web_driver_location
        self._query_text = query
        self._scoring_method = scoring_method
        
        self._db_api = APIClient(base_url=database_base_url)

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
        self._front_metadata_dict = dict()
        self._link_version_dict = dict()

        self._shared_downloaded_page_count = 0
        self._lock_downloaded_page_count = threading.Lock()

        self._shared_robots_info = {}
        self._shared_last_access = {}
        self._lock_website_access_info = threading.Lock()

        self._site_table_data = {}

        # classla.download("sl")
        # self.classla_nlp = classla.Pipeline("sl")

        # robots.txt info
        for domain in self._domains:
            robots_url = domain.rstrip("/") + "/robots.txt"
            try:
                r = requests.get(robots_url, timeout=5)
                rp = RobotExclusionRulesParser()
                rp.parse(r.text)
                
                # print(rp.sitemaps)
                sitemap_r = requests.get(rp.sitemaps[0], timeout=5)
                # print(sitemap_r.text)
                delay = rp.get_crawl_delay(self._crawler_id)

                if delay is None:
                    delay = rp.get_crawl_delay("*")

                if delay is None:
                    delay = default_crawl_delay

                self._shared_robots_info[domain] = {
                    "info": rp,
                    "delay": delay
                }

                self._site_table_data[domain] = {
                    "robots_content": r.text,
                    "sitemap_content": sitemap_r.text,
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

        if scoring_method == 'BERT':
            self._query_embed = embed_BERT(self._query_text)

        # initialize queue
        for seed in seed_urls:
            self._shared_crawling_front.put((0, (seed, 0, -1)))






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
            # try:
            #     # print(url)
            #     # print("looking for proad")
            #     #self._logger.debug(worker_driver.page_source)
            #     proad = worker_driver.find_element(By.ID, "proad")
            #     # print("found proad?")
            #     # print(BeautifulSoup(worker_driver.page_source))
            #     ActionChains(worker_driver).scroll_to_element(proad).perform()
            #     wait = WebDriverWait(worker_driver, timeout=3)
            #     # print("trying to scroll to proad-stat")
            #     wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "proad-stat")))
                
            # except:
            #     self._logger.info("proad-stat container doesn't exist")
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
                priority, (url, link_version, from_page_id) = self._shared_crawling_front.get(timeout=3)
            except:
                break
            
            # throw out links with old priority score
            if url in self._link_version_dict and link_version < self._link_version_dict[url]:
                self._logger.info(f"Old link (v{link_version}): {url}")
                self._shared_crawling_front.task_done()
                continue

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

            rb : RobotExclusionRulesParser = self._shared_robots_info[domain]["info"]
            if not rb.is_allowed(rb.user_agent, url):
                continue

            self._respect_crawl_delay(domain)
            html = self._fetch_page(worker_web_driver, url)

            # invalid response - no body
            if not html:
                self._shared_crawling_front.task_done()
                continue
            
            # add to visited set
            with self._lock_visited_urls:
                self._shared_visited_urls.add(url)


            # save to DB
            page_id = save_page_to_db(self._logger, url, html, from_page_id, self._db_api)
            if page_id == -1:
                self._logger.warning(f"Error saving html contents of {url} to DB")

            # process links
            website_data = parse_website_content(html, url, rb)
            website_urls = list(website_data.keys())
            
            self._logger.info(f"[{worker_name}] Crawled:   {url}")
            self._logger.info(f"  - Found {len(website_urls)} links")
            for link in website_urls:
                # link, tag = url_data[0], url_data[1] # dummy simple unclean dat
                # link_norm = normalize_url(link) # link norm is already called in parse website content
                #print(link_norm)
                #rint(website_data[link])
                link_version = 0

                with self._lock_visited_urls:
                    if link in self._shared_visited_urls:
                        continue

                    # add metadata to list of metadata for the found link
                    if link in self._front_metadata_dict:
                        self.append_metadata(link, website_data[link])
                    else:
                        self._front_metadata_dict[link] = [website_data[link]]

                    # increment and store the latest version of the link in the priority queue, so older ones are thrown out
                    if link in self._link_version_dict:
                        self._link_version_dict[link] += 1
                    else:
                        self._link_version_dict[link] = 0

                    link_version = self._link_version_dict[link]
                
                priority = 0
                if self._scoring_method == 'BERT':
                    priority = priority_score_BERT(self._logger, html, link, self._front_metadata_dict[link], self._query_embed)
                elif self._scoring_method == 'BOW':
                    priority = priority_score_BOW(self._logger, html, link, self._front_metadata_dict[link], self._query_text)
                
                # self._logger.debug(f"Priority: {priority}, link: {link}")
                self._shared_crawling_front.put((-priority, (link, link_version, page_id))) # minus priority, because priority queue returns smallest priority

            with self._lock_visited_urls:
                for i in range(5):
                    element = self._shared_crawling_front.queue[i]
                    self._logger.debug(f"[Top {i+1}] Priority: {-element[0]}, link: {element[1][0]}")
            self._shared_crawling_front.task_done()

        worker_web_driver.quit()

    def append_metadata(self, link, metadata):
        self._front_metadata_dict[link].append(metadata)

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
    # seed = "https://www.24ur.com/novice/tujina"

    crawler = WebCrawler24Ur(
        seed_urls=[seed],
        max_pages=10,
        worker_count=1,
        log_to_stdout=True,
        logging_file='./crawler.log',
        logging_level='DEBUG',
        query="Olimpijske igre."
    )

    crawler.crawl()