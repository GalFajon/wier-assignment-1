# wier-assignment-1

Assignment 1 webcrawler repo.

## Project structure:

* exercises-notebook contains the notebooks from učilnica, dependencies are in requirements.txt
* src contains source code, current plan is that the crawler will be a docker-compose project:

  * db -> a database with a persistent storage volume
  * server -> rest API in flask
  * client -> container that crawls and updates the API, made to be ran in parallel

---

## TODO
### BASIC
* [x] multithreading - multiple workers in parallel, number of workers is parameter
* [x] preferential strategy - EXPLAINATION IN REPORT ABOUT STRATEGY
* [ ] check and FULLY respect robots.txt - respect the commands User-agent, Allow, Disallow, Crawl-delay and Sitemap
* [ ] follow ethics and do not send request to the same server more often than one request in 5 seconds (not only domain but also IP!)
* [ ] store canonicalized URLs only in DB
* [ ] duplicate web pages during crawling
* [ ] 

### WEBSITE RENDERING / PARSING
* [ ] when parsing links, include links from href attributes and onclick Javascript events (e.g. location.href or document.location). Be careful to correctly extend the relative URLs before adding them to the frontier
* [ ] detect images on a web page only based on img tag, where the src attribute points to an image URL
* [ ] detect the relevance of each link to your domain and crawl more relevant links first
* [ ] download HTML content only (and PDF where required for the domain) - List all other content (.doc, .docx, .ppt and .pptx) in the page_data table - there is no need to populate data field (i.e. binary content), In case you put a link into a frontier and identify content as a binary source, you can just set its page_type to BINARY. The same holds for the image data
* [ ] 

### DATABASE
* [ ] table site contains web site specific data - DOMAIN - 24ur.com
* [ ] 
