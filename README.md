# wier-assignment-1

Assignment 1 webcrawler repo.

## Project structure:
* exercises-notebook contains the notebooks from učilnica, dependencies are in requirements.txt
* pa1 contains assignment 1 crawler implementation
---

## Assignment 1 TODO
### BASIC
* [x] The crawler needs to be implemented with multiple workers that retrieve different web pages in parallel
  * [x] The number of workers should be a parameter when starting the crawler.
* [x] The frontier strategy needs to follow the preferential strategy.
* [x] Check and respect the robots.txt file for each domain if it exists.
  * [x] Correctly respect the commands User-agent, Allow, Disallow, Crawl-delay and Sitemap.
* [ ] Also make sure that you follow ethics and do not send request to the same server more often than one request in 5 seconds (not only domain but also IP!).
* [x] In your implementation you must set the User-Agent field of your bot to fri-wier-NAME_OF_YOUR_GROUP (fri-wier-D)

### WEBSITE PARSING
* [x] When parsing links, include links from href attributes and onclick Javascript events (e.g. location.href or document.location). Be careful to correctly extend the relative URLs before adding them to the frontier.
* [x] Detect images on a web page only based on img tag, where the src attribute points to an image URL.
* [x] Detect the relevance of each link to your domain and crawl more relevant links first. This will also help with your subsequent programming assignments as you will have more relavant pages to work with.
* [x] Download HTML content only (and PDF where required for the domain).
  * [x] List all other content (.doc, .docx, .ppt and .pptx) in the page_data table - there is no need to populate data field (i.e. binary content)
  * [x] In case you put a link into a frontier and identify content as a binary source, you can just set its page_type to BINARY. The same holds for the image data.

### DATABASE STORING
* [x] In a database store canonicalized URLs only!
* [ ] During crawling you need to detect duplicate web pages.
  * [ ] extend the database with a hash, otherwise you need compare exact HTML code
  * [ ] The duplicate page should not have set the html_content value and should be linked to a duplicate version of a page.
* [x] If your crawler gets a URL from a frontier that has already been parsed, this is not treated as a duplicate. In such cases there is no need to re-crawl the page, just add a record into to the TABLE LINK accordingly.

### REPORT
* [ ] In the report explain how your preferrential strategy is implemented.
