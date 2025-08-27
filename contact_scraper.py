import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import re
import json
import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from urllib.parse import urljoin, urlparse

import requests
import pandas as pd
import tldextract
from bs4 import BeautifulSoup
from openai import OpenAI

# =============================================================================
# Minimal logging (prints + UI log list)
# =============================================================================
LOG_LEVEL = os.getenv("SCRAPER_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("contact_scraper")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

def _log(logs: Optional[List[str]], msg: str):
    print(msg, flush=True)
    logger.info(msg)
    if logs is not None:
        logs.append(msg)

# =============================================================================
# Config & constants
# =============================================================================
UA = "methane-leaks/2.0 (contact-scraper)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"})

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{6,}\d)")

SOCIAL_DOMAINS = ("facebook.com", "linkedin.com", "instagram.com", "twitter.com", "x.com",
                  "tiktok.com", "youtube.com")

HIGH_VALUE_PATHS = [
    "/", "/contact", "/contact-us", "/about", "/about-us",
    "/investors", "/investor-relations", "/ir",
    "/press", "/media", "/newsroom",
    "/sustainability", "/esg", "/csr",
    "/privacy", "/privacy-policy", "/terms", "/terms-of-use", "/legal",
    "/impressum", "/imprint", "/empresa", "/contacto", "/quienes-somos", "/kontakt",
]

# Optional Google CSE
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_CX  = os.getenv("GOOGLE_CX", "").strip()

OPENAI_SEARCH_MODEL = os.getenv("OPENAI_SEARCH_MODEL", "gpt-4o-mini").strip()

# NOTE: Do NOT read SCRAPER_SITE_OVERRIDES here (import time). We read it dynamically in _load_overrides().

# =============================================================================
# Small helpers
# =============================================================================
def _ensure_scheme(url: str) -> str:
    if not url:
        return url
    if not url.lower().startswith(("http://", "https://")):
        return "https://" + url
    return url

def _is_social(url: str) -> bool:
    if not url:
        return False
    host = urlparse(url).netloc.lower()
    return any(s in host for s in SOCIAL_DOMAINS)

def _domain_from_url(url: str) -> str:
    parts = tldextract.extract(url or "")
    if not parts.domain:
        return ""
    return ".".join([p for p in [parts.domain, parts.suffix] if p])

def _collect_contacts_from_html(html_text: str) -> Tuple[List[str], List[str]]:
    emails = {e.strip().strip(".,;").lower() for e in EMAIL_RE.findall(html_text or "")}
    phones = {p.strip() for p in PHONE_RE.findall(html_text or "")}
    return sorted(emails), sorted(phones)

def _best_email(emails: List[str], domain: str) -> str:
    """Pick a sensible email: same-domain preferred, otherwise role-ish, else first."""
    if not emails:
        return ""
    if domain:
        domain_hits = [e for e in emails if domain in e]
        if domain_hits:
            for e in domain_hits:
                if "noreply" not in e and "no-reply" not in e:
                    return e
            return domain_hits[0]
    role_order = ["environment", "env@", "compliance", "operations", "ops@", "sustainability",
                  "esg", "media", "press", "investor", "ir@"]
    for role in role_order:
        for e in emails:
            if role in e:
                return e
    return emails[0]

# =============================================================================
# Overrides & a single optional search provider
# =============================================================================
def _load_overrides(logs: Optional[List[str]]) -> Dict[str, str]:
    """
    Load manual name→site mappings from site_overrides.json.
    Priority:
      1) SCRAPER_SITE_OVERRIDES env var (evaluated at call time so tests/monkeypatch work)
      2) ./site_overrides.json in current working directory
    """
    # Read env dynamically on every call (fixes pytest monkeypatch + hot reload)
    env_path = os.getenv("SCRAPER_SITE_OVERRIDES", "").strip()
    path = env_path if env_path else ""

    if not path:
        cwd_path = os.path.join(os.getcwd(), "site_overrides.json")
        path = cwd_path if os.path.exists(cwd_path) else ""

    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _log(logs, f"Overrides loaded from {path} ({len(data)} entries).")
            return {k.lower(): v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
        except Exception as e:
            _log(logs, f"Could not load overrides from {path}: {e}")
            return {}
    else:
        _log(logs, "No overrides file found (optional).")
        return {}

def _match_override(name: str, overrides: Dict[str, str]) -> str:
    nm = (name or "").lower()
    for key, site in overrides.items():
        if key and key in nm:
            return site
    return ""


def _openai_web_search_first_result(query: str, logs: Optional[List[str]], timeout: int = 20) -> Optional[str]:
    """Use OpenAI's web_search tool to locate the official website."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        _log(logs, "OPENAI_API_KEY not set — skipping OpenAI search.")
        return None
    try:
        _log(logs, f"[OpenAI] query={query}")
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=OPENAI_SEARCH_MODEL or "gpt-4o-mini",
            tools=[{"type": "web_search", "search_context_size": "low"}],
            input=f"Find the official homepage for {query}. Return only the URL.",
        )
        text = (getattr(resp, "output_text", "") or "").strip()
        match = re.search(r"https?://[^\s]+", text)
        if match:
            url = match.group(0).strip("'\".,)")
            if not _is_social(url):
                return url
    except Exception as e:
        _log(logs, f"[OpenAI] error: {e}")
    return None

def _google_cse_first_result(query: str, logs: Optional[List[str]], timeout: int = 10) -> Optional[str]:
    if not (GOOGLE_KEY and GOOGLE_CX):
        if GOOGLE_KEY and not GOOGLE_CX:
            _log(logs, "Google API key present but GOOGLE_CX missing — skipping Google CSE.")
        elif GOOGLE_CX and not GOOGLE_KEY:
            _log(logs, "GOOGLE_CX present but GOOGLE_API_KEY missing — skipping Google CSE.")
        else:
            _log(logs, "Google CSE keys not configured — skipping search.")
        return None
    try:
        _log(logs, f"[Google CSE] query={query}")
        r = SESSION.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": GOOGLE_KEY, "cx": GOOGLE_CX, "q": query, "num": 5},
            timeout=timeout
        )
        if r.status_code != 200:
            _log(logs, f"[Google CSE] non-200: {r.status_code}")
            return None
        items = (r.json().get("items") or [])
        for it in items:
            url = it.get("link")
            if url and not _is_social(url):
                return url
    except Exception as e:
        _log(logs, f"[Google CSE] error: {e}")
    return None

def _duckduckgo_first_result(query: str, logs: Optional[List[str]], timeout: int = 10) -> Optional[str]:
    """Simple DuckDuckGo search using the public HTML endpoint."""
    try:
        _log(logs, f"[DuckDuckGo] query={query}")
        r = SESSION.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=timeout,
        )
        if r.status_code != 200:
            _log(logs, f"[DuckDuckGo] non-200: {r.status_code}")
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select("a.result__a"):
            href = a.get("href")
            if href and not _is_social(href):
                return href
    except Exception as e:
        _log(logs, f"[DuckDuckGo] error: {e}")
    return None

def _search_first_result(query: str, logs: Optional[List[str]], provider: str = "auto", timeout: int = 10) -> Optional[str]:
    """Return first result from configured provider (OpenAI, Google or DuckDuckGo)."""
    provider = (provider or "auto").lower()
    # Determine search order
    searchers: List[Tuple[str, Callable[..., Optional[str]]]] = []
    if provider in ("openai", "auto"):
        searchers.append(("openai", lambda q: _openai_web_search_first_result(q, logs, timeout)))
    if provider in ("google", "auto"):
        searchers.append(("google", lambda q: _google_cse_first_result(q, logs, timeout)))
    if provider in ("duckduckgo", "auto"):
        searchers.append(("duckduckgo", lambda q: _duckduckgo_first_result(q, logs, timeout)))

    for name, func in searchers:
        url = func(query)
        if url:
            if name == "google":
                _log(logs, f"Resolved via Google CSE: {url}")
            elif name == "openai":
                _log(logs, f"Resolved via OpenAI web search: {url}")
            else:
                _log(logs, f"Resolved via DuckDuckGo: {url}")
            return url
    return None

# =============================================================================
# Website resolution (simple & deterministic)
# =============================================================================
def resolve_website(
    name: str,
    country: str,
    lat: Optional[float],
    lon: Optional[float],
    given_website: str,
    logs: Optional[List[str]],
    search_provider: str = "auto",
) -> str:
    """
    1) Use given website if present.
    2) Try manual overrides (site_overrides.json or env path).
    3) Try search provider (OpenAI web search, Google CSE or DuckDuckGo):
       "<name> <country>" then "<name>".
    4) Otherwise: return "" and let the UI show 'add mapping'.
    """
    site = (given_website or "").strip()
    if site:
        _log(logs, f"Website provided by candidate: {site}")
        return _ensure_scheme(site)

    ov = _load_overrides(logs)
    matched = _match_override(name, ov)
    if matched:
        _log(logs, f"Override matched: {matched}")
        return _ensure_scheme(matched)

    q1 = f"{name} {country}".strip()
    site = _search_first_result(q1, logs, provider=search_provider) or _search_first_result(name, logs, provider=search_provider)
    if site:
        return _ensure_scheme(site)

    _log(logs, "No website resolved (consider adding to site_overrides.json).")
    return ""

# =============================================================================
# Crawl contacts from a site (small, robust)
# =============================================================================
def _http_get(url: str, timeout: int, retries: int, logs: Optional[List[str]]) -> Optional[str]:
    url = _ensure_scheme(url)
    for attempt in range(retries + 1):
        try:
            _log(logs, f"GET {url} (attempt {attempt+1})")
            r = SESSION.get(url, timeout=timeout, allow_redirects=True)
            if 200 <= r.status_code < 300 and (r.text or "").strip():
                return r.text
            _log(logs, f"Non-2xx: {r.status_code} for {url}")
        except requests.RequestException as e:
            _log(logs, f"Request error on {url}: {e}")
        time.sleep(0.25 * (attempt + 1))
    return None

def crawl_for_contacts(
    base_site: str,
    timeout: int,
    max_pages: int,
    retries: int,
    logs: Optional[List[str]]
) -> Tuple[List[str], List[str], List[str]]:
    visited: List[str] = []
    emails_all, phones_all = set(), set()
    seen = set()
    queue = [urljoin(base_site, p) for p in HIGH_VALUE_PATHS]

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if not url or url in seen:
            continue
        seen.add(url)
        html = _http_get(url, timeout=timeout, retries=retries, logs=logs)
        if not html:
            continue
        visited.append(url)
        ems, phs = _collect_contacts_from_html(html)
        emails_all.update(ems)
        phones_all.update(phs)

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a[href]"):
            href = a.get("href") or ""
            lower = href.lower()
            if any(k in lower for k in ["contact", "kontakt", "impressum", "imprint", "legal", "privacy", "terms",
                                        "investor", "media", "press", "contacto", "empresa"]):
                nxt = urljoin(url, href)
                if nxt not in seen and len(visited) + len(queue) < max_pages:
                    queue.append(nxt)
                    break

    return sorted(emails_all), sorted(phones_all), visited

# =============================================================================
# Public API (same signatures as before)
# =============================================================================
def scrape_contacts_for_candidate(
    cand: Dict[str, Any],
    timeout: int = 12,
    max_pages: int = 8,
    retries: int = 1,
    logs: Optional[List[str]] = None,
    enrich_missing_website: bool = True,
    search_provider: str = "auto",
) -> Dict[str, Any]:
    name = cand.get("name") or "(unnamed)"
    country = cand.get("country") or ""
    lat = cand.get("lat")
    lon = cand.get("lon")
    given_site = cand.get("website") or ""

    _log(logs, f"=== Candidate: {name}")

    site = resolve_website(name, country, lat, lon, given_site, logs, search_provider=search_provider) if enrich_missing_website else (given_site or "")
    site = _ensure_scheme(site)

    row = {
        "Candidate": name,
        "Website": site,
        "Best Email": "",
        "Best Phone": "",
        "All Emails": "",
        "All Phones": "",
        "Pages Checked": "",
        "Lat": cand.get("lat"),
        "Lon": cand.get("lon"),
        "Distance_km": cand.get("distance_km"),
        "Source": cand.get("source"),
        "Error": "",
    }

    if not site:
        _log(logs, "No website — POC suggests adding to site_overrides.json.")
        return row

    domain = _domain_from_url(site)
    emails, phones, pages = crawl_for_contacts(site, timeout=timeout, max_pages=max_pages, retries=retries, logs=logs)

    row["All Emails"] = ", ".join(emails)
    row["All Phones"] = ", ".join(phones)
    row["Pages Checked"] = ", ".join(pages)
    row["Best Email"] = _best_email(emails, domain) if emails else ""
    row["Best Phone"] = phones[0] if phones else ""

    _log(logs, f"Collected {len(emails)} emails, {len(phones)} phones.")
    return row


def scrape_contacts_bulk(
    cands: List[Dict[str, Any]],
    debug: bool = False,
    timeout: int = 12,
    max_pages: int = 8,
    retries: int = 1,
    enrich_missing_website: bool = True,
    search_provider: str = "auto",  # kept for compatibility; not used
    progress_cb: Optional[Callable[[float], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    logs: List[str] = []
    if debug:
        logger.setLevel(logging.DEBUG)
        _log(logs, "Debug logging enabled.")

    if not isinstance(cands, list) or not cands:
        _log(logs, "No candidates passed to scraper.")
        df = pd.DataFrame([{
            "Candidate": "(none)", "Website": "", "Best Email": "",
            "Best Phone": "", "All Emails": "", "All Phones": "",
            "Pages Checked": "", "Lat": "", "Lon": "", "Distance_km": "", "Source": "", "Error": "No candidates"
        }])
        if progress_cb: progress_cb(1.0)
        if log_cb: log_cb("\n".join(logs[-50:]))
        return df, logs

    rows = []
    total = len(cands)
    for i, c in enumerate(cands):
        try:
            _log(logs, f"[{i+1}/{total}] Starting candidate: {c.get('name') or '(unnamed)'}")
            if log_cb: log_cb("\n".join(logs[-80:]))

            row = scrape_contacts_for_candidate(
                c, timeout=timeout, max_pages=max_pages, retries=retries,
                logs=logs, enrich_missing_website=enrich_missing_website,
                search_provider=search_provider
            )
            rows.append(row)
        except Exception as e:
            _log(logs, f"ERROR in candidate: {e}")
            rows.append({
                "Candidate": c.get("name") or "(unnamed)", "Website": c.get("website") or "",
                "Best Email": "", "Best Phone": "", "All Emails": "", "All Phones": "",
                "Pages Checked": "", "Lat": c.get("lat"), "Lon": c.get("lon"),
                "Distance_km": c.get("distance_km"), "Source": c.get("source"),
                "Error": str(e)
            })

        if progress_cb:
            progress_cb((i + 1) / total)
        if log_cb:
            log_cb("\n".join(logs[-80:]))

    df = pd.DataFrame(rows)
    df["__has_email__"] = df["Best Email"].apply(lambda x: 0 if (isinstance(x, str) and "@" in x) else 1)
    sort_cols = [c for c in ["__has_email__", "Distance_km", "Candidate"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last").drop(columns=["__has_email__"], errors="ignore").reset_index(drop=True)

    _log(logs, f"Scraping complete. {len(df)} rows.")
    if log_cb:
        log_cb("\n".join(logs[-80:]))

    return df, logs


def scrape_contacts_bulk_compat(cands: List[Dict[str, Any]]):
    df, _ = scrape_contacts_bulk(cands, debug=False)
    return df