import re
from lxml import etree
from lxml import html as lxml_html
from typing import Any


def parse_html(html_content: str) -> dict[str, Any]:
	#print("")
	try:
		tree = lxml_html.fromstring(html_content)
	except (etree.ParserError, ValueError):
		return {
			"title": "",
			"article-content": "",
		}

	raw_title = tree.xpath("normalize-space(string(//meta[@property='og:title']/@content))")
	if not raw_title:
		raw_title = tree.xpath("normalize-space(string(//title))")
	title = re.sub(r"\s*\|\s*24ur\.com\s*$", "", raw_title, flags=re.IGNORECASE).strip()

	article_nodes = tree.xpath("//*[@id='article-body']")
	if not article_nodes:
		return {
			"title": title,
			"article-content": "",
		}
	
		

	article_node = article_nodes[0]
	for removable in article_node.xpath(".//script|.//style|.//noscript|.//img|.//figure|.//svg|.//picture"):
		parent = removable.getparent()
		if parent is not None:
			parent.remove(removable)

	article_text = re.sub(r"\s+", " ", article_node.text_content()).strip()

	# extract sumamry by detecting the main article image, and moving along hierarchy to the summary <p> element
	summary = tree.xpath("//picture[@tabindex=0 and @class='media-object']//parent::div//parent::div//parent::div/p//text()")
	if summary:
		article_text = f'{summary[0]} {article_text}'

	return {
		"title": title,
		"article-content": article_text,
	}
