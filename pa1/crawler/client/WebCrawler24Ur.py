import logging
import sys
import time
from typing import List
from bs4 import BeautifulSoup
import requests
import threading
from queue import PriorityQueue
import ppdeep

from urllib.parse import urlsplit
from robotexclusionrulesparser import RobotExclusionRulesParser
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.support import expected_conditions as EC

from utils.priority_scoring import embed_BERT, BERT_score_batch  # type: ignore
from utils.website_parsing import parse_website_content # type: ignore
from utils.api_client import APIClient
from utils.url_cleaning import canonicalize_url
from utils.database_saving import get_site_id_or_create_site, save_frontier_page_to_db, save_page_to_db, save_link

class WebCrawler24Ur:

    def __init__(
        self,
        seed_urls: List[str],
        crawler_id: str = "fri-wier-D",
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
        self._default_crawl_delay = default_crawl_delay
        
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

        self._get_robots_data()
        
        if scoring_method == 'BERT':
            self._query_embed = embed_BERT(self._query_text)

        # initialize queue
        for seed in seed_urls:
            self._shared_crawling_front.put((0, (seed, 0, -1, -1)))


    def _get_robots_data(self):

        # robots.txt info
        site_data = []

        for domain in self._domains:
            robots_url = domain.rstrip("/") + "/robots.txt"

            normalized_url = canonicalize_url(domain)
            parsed = urlsplit(normalized_url)
            canon_domain = parsed.netloc

            try:
                r = requests.get(robots_url, timeout=5)

                rp = RobotExclusionRulesParser()
                rp.parse(r.text)

                sitemap_content = None
                sitemap_url = None

                if rp.sitemaps:
                    sitemap_url = rp.sitemaps[0]
                    sitemap_r = requests.get(sitemap_url, timeout=5)
                    sitemap_content = sitemap_r.text

                delay = rp.get_crawl_delay(self._crawler_id)
                if delay is None:
                    delay = rp.get_crawl_delay("*")
                if delay is None:
                    delay = self._default_crawl_delay

                self._shared_robots_info[domain] = {
                    "info": rp,
                    "delay": delay
                }

                site_data.append({
                    "domain": canon_domain,
                    "robots_content": r.text,
                    "sitemap_content": sitemap_content
                })

            except Exception:
                self._shared_robots_info[domain] = {
                    "info": None,
                    "delay": self._default_crawl_delay
                }

                site_data.append({
                    "domain": canon_domain,
                    "robots_content": '',
                    "sitemap_content": ''
                })

            self._shared_last_access[domain] = time.time()

        self._logger.info("MAIN - INITIALIZED CRAWLER")
        delay_info = {d: v["delay"] for d, v in self._shared_robots_info.items()}
        self._logger.debug(f"Crawl delays: {delay_info}")

        #save sites to DB
        for site in site_data:
            id = get_site_id_or_create_site(self._logger, site, self._db_api)



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
        if not domain_info:
            return False

        rp = domain_info['info']
        if rp is not None and not rp.is_allowed(self._crawler_id, url):
            self._logger.info('_valid_url - URL:', url, "is NOT allowed by robots.txt")
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
                priority, (url, link_version, from_page_id, crawling_page_front_id) = self._shared_crawling_front.get(timeout=3)
            except:
                break
            
            # throw out links with old priority score
            if url in self._link_version_dict and link_version < self._link_version_dict[url]:
                self._logger.debug(f"Old link (v{link_version}): {url}")
                self._shared_crawling_front.task_done()
                continue

            # validate url
            if not self._valid_url(url):
                self._shared_crawling_front.task_done()
                continue

            # process url
            url_parts = urlsplit(url)
            domain = url_parts.scheme + "://" + url_parts.netloc

            rb : RobotExclusionRulesParser = self._shared_robots_info[domain]["info"]
            if not rb.is_allowed(rb.user_agent, url):
                continue
            
            self._logger.info(f"_deploy_crawl_worker - Beginning to process page: {url} with domain {domain} pulled with prio {priority}")
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
            page_id = save_page_to_db(self._logger, url, html, from_page_id, crawling_page_front_id, self._db_api)
            if page_id == -1:
                self._logger.warning(f"_deploy_crawl_worker - Error saving html contents of {url} to DB")
                continue


            # process links
            website_data = parse_website_content(html, url, rb)
            website_urls = list(website_data.keys())
            
            self._logger.info(f"_deploy_crawl_worker - [{worker_name}] Crawled:   {url}")
            self._logger.info(f"  - Found {len(website_urls)} links")

            # url scoring inside page
            candidates = []
            for link in website_urls:
                with self._lock_visited_urls:
                    if link in self._shared_visited_urls:
                        continue

                    if link in self._front_metadata_dict:
                        self.append_metadata(link, website_data[link])
                    else:
                        self._front_metadata_dict[link] = [website_data[link]]

                    if link in self._link_version_dict:
                        self._link_version_dict[link] += 1
                    else:
                        self._link_version_dict[link] = 0

                    link_version = self._link_version_dict[link]

                candidates.append({
                    "link": link,
                    "version": link_version,
                    "metadata": self._front_metadata_dict[link]
                })

            if candidates:
                scores = BERT_score_batch(self._logger, candidates, self._query_embed)
                scores = scores.cpu().numpy()

            for i, candidate in enumerate(candidates):
                
                priority = float(-scores[i])
                link = candidate['link']
                link_version = candidate['version']

                frontier_page_entry = {
                    "priority": priority,
                    "url": link
                }

                frontier_page_id = save_frontier_page_to_db(self._logger, frontier_page_entry, self._db_api)
                if frontier_page_id == None:
                    continue
                
                save_link(self._logger, page_id, frontier_page_id, self._db_api)
                self._shared_crawling_front.put((priority, (link, link_version, page_id, frontier_page_id)))


            with self._lock_visited_urls:
                for i in range(min(len(self._shared_crawling_front.queue), 5)):
                    element = self._shared_crawling_front.queue[i]
                    #self._logger.debug(f"[Top {i+1}] Priority: {-element[0]}, link: {element[1][0]}")
            self._shared_crawling_front.task_done()

            # end if reached page count max
            with self._lock_downloaded_page_count:
                if self._shared_downloaded_page_count >= self._max_pages:
                    self._shared_crawling_front.task_done()
                    break
                self._shared_downloaded_page_count += 1

                if self._shared_downloaded_page_count % 25 == 0:
                    self._logger.info(f"_deploy_crawl_worker - CRAWLED {self._shared_downloaded_page_count} PAGES SO FAR!")

        worker_web_driver.quit()

    def append_metadata(self, link, metadata):
        self._front_metadata_dict[link].append(metadata)

    def crawl(self):
        self._logger.info(f"MAIN - Beginning crawl")

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

    # seed = "https://www.24ur.com/"

    seed = "https://www.24ur.com/"
    #seed = 'https://www.sdl.si/bivanje-v-sdl/domski-red/'
    #print(ppdeep.compare("384:TmYpaRqjmWQwzbymqP2UuPcEBc2CZNXtPHGT4K/GwHkQ7wP/TJy6JUqPcUmYmTE1:TmYpaRqjFbbMukWc2StvmYmTEIAlo/P0", "384:TmYpHCi5mWQrqZymqP2UuPcEBc2WtULtPHGT4K/GwHkQ7wP/TJy6JUqPcUmYmTE1:TmYpHCi5FRZMukWc21LvmYmTEIAlo/P0"))

    crawler = WebCrawler24Ur(
        seed_urls=[seed],
        max_pages=15,
        worker_count=1,
        log_to_stdout=True,
        logging_file='./crawler.log',
        logging_level='DEBUG',
        query="Vojna med Rusijo in Ukrajino."
    )


    crawler.crawl()