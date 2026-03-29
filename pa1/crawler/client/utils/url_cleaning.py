from urllib.parse import urlparse, urlunparse, parse_qsl, parse_qs, urlencode, ParseResult
import posixpath
import tldextract

extractor = tldextract.TLDExtract(cache_dir="./cache/tld_cache")

PROTOCOL_SCHEME = 'https'

ALLOWED_PARAMS = {
    "e",
}

REDIRECT_DOMAINS = {
    "go-ctr-tracker.pub.24ur.si"
}


def link_domain(url):
    ext = extractor(url)
    domain = f"{ext.domain}.{ext.suffix}"
    return f"https://{domain}"


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)

    if parsed.netloc in REDIRECT_DOMAINS:
        qs = dict(parse_qsl(parsed.query))
        redirect_url = qs.get("redir") or qs.get("amp;redir")
        if redirect_url:
            parsed = urlparse(redirect_url)

    ext = extractor(parsed.geturl())
    netloc = f"{ext.domain}.{ext.suffix}"

    path = parsed.path or "/"
    path = posixpath.normpath(path)

    if path == ".":
        path = "/"
    elif path != "/":
        path = path.rstrip("/")

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [(k, v) for (k, v) in query_pairs if k in ALLOWED_PARAMS]

    filtered.sort()
    query = urlencode(filtered, doseq=True)

    canonicalized = urlunparse((PROTOCOL_SCHEME, netloc, path, "", query, ""))
    return canonicalized



def normalize_url(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc == "go-ctr-tracker.pub.24ur.si":
        redirect_url = parse_qs(parsed_url.query)["amp;redir"][0]
        #print(redirect_url)
        parsed_url = urlparse(redirect_url)
        #new_redirect_result = ParseResult(parsed_redirect.scheme, parsed_redirect.netloc, parsed_redirect.path, "", "", "")

    new_url_result = ParseResult(parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", "", "")
    unparsed_url = urlunparse(new_url_result)
    #print(unparsed_url)
    return unparsed_url

if __name__ == "__main__":
    #normalize_url("https://go-ctr-tracker.pub.24ur.si/rec/JS_1/c/1/48ed56fa-f1af-448f-9e82-e1e54c97222c/1.gif?articleId=4422781&amp;at=1773862800&amp;mobile=0&amp;redir=https%3A%2F%2Fwww.24ur.com%2Fnovice%2Fslovenija%2Fgolob-na-visku-kampanje-z-ostrimi-besedami-proti-politicnim-nasprotnikom.html%3Futm_source%3DProAd%26utm_medium%3D24ur%26utm_content%3DProAd_24ur__%26utm_campaign%3DProAd&amp;source=vector&amp;sig=3ddbd03ce6de900ff91a61393b722bb611a6e57722f3ad0d335b1448d56b5e54")

    url1 = 'https://24ur.com/novice/svet/preiskovalci-zakljucili-malezijsko-letalo-nad-ukrajino-sestrelila-ruska-vojska.html?q=letalo%2C+raketa&sort=score&t=article'
    url2 = 'https://24ur.com/spored/kanal/popkino?e=12817738'

    print(canonicalize_url(url1))
    print(canonicalize_url(url2))

