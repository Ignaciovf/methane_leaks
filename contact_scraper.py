import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import re
import time
import html
import json
import logging
import tldextract
import requests
import pandas as pd

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote_plus
from typing import List, Dict, Any, Tuple, Optional, Callable

# -----------------------
# Logging
# -----------------------
LOG_LEVEL = os.getenv("SCRAPER_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("contact_scraper")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

UA = "methane-leaks/1.0 (contact-scraper)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"})

EMAIL_RE = re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.I)
PHONE_RE = re.compile(r'(\+?\d[\d\s().-]{6,}\d)')

SOCIAL_DOMAINS = ("facebook.com", "linkedin.com", "instagram.com", "twitter.com", "x.com", "tiktok.com", "youtube.com")
CANDIDATE_PATHS = [
    "/", "/contact", "/contact-us", "/about", "/about-us", "/imprint", "/legal",
    "/empresa", "/contacto", "/quienes-somos", "/impressum", "/kontakt"
]


def _norm_url(url: str) -> str:
    return (url or "").split("#")[0].strip()


def _ensure_scheme(url: str) -> str:
    if not url:
        return url
    if not url.lower().startswith(("http://", "https://")):
        return "https://" + url
    return url


def _domain_from_url(url: str) -> str:
    parts = tldextract.extract(url or "")
    if not parts.domain:
        return ""
    return ".".join([p for p in [parts.domain, parts.suffix] if p])


def _is_social(url: str) -> bool:
    if not url:
        return False
    host = urlparse(url).netloc.lower()
    return any(s in host for s in SOCIAL_DOMAINS)


def _get(url: str, timeout: int = 12, retries: int = 1, logs: List[str] = None) -> Optional[requests.Response]:
    url = _ensure_scheme(url)
    for attempt in range(retries + 1):
        try:
            if logs is not None:
                logs.append(f"GET {url} (attempt {attempt+1})")
            resp = SESSION.get(url, timeout=timeout, allow_redirects=True)
            if 200 <= resp.status_code < 300:
                return resp
            if logs is not None:
                logs.append(f"Non-2xx: {resp.status_code} for {url}")
        except requests.RequestException as e:
            if logs is not None:
                logs.append(f"Request error on {url}: {e}")
        time.sleep(0.35 * (attempt + 1))
    return None


def _collect_contacts_from_html(html_text: str) -> Tuple[List[str], List[str]]:
    emails = set(EMAIL_RE.findall(html_text or ""))
    phones = set(PHONE_RE.findall(html_text or ""))
    emails = {e.strip().strip(".,;") for e in emails}
    phones = {p.strip() for p in phones}
    return sorted(emails), sorted(phones)


def _first_contactish_link(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        lower = href.lower()
        if any(k in lower for k in ["contact", "kontakt", "impressum", "imprint", "legal", "empresa", "contacto"]):
            return urljoin(base_url, href)
    return None


def _best_email(emails: List[str], domain: str) -> str:
    if not emails:
        return ""
    if domain:
        domain_emails = [e for e in emails if e.lower().endswith("@" + domain) or domain in e.lower()]
        if domain_emails:
            return domain_emails[0]
    return emails[0]


def _sanitize_candidate_website(website: str) -> str:
    if not website:
        return ""
    website = _ensure_scheme(website.strip())
    if _is_social(website):
        return ""
    return website


# -----------------------
# Enrichment: OSM + DuckDuckGo
# -----------------------
def _nominatim_search(name: str, logs: List[str], timeout: int = 10) -> Optional[dict]:
    try:
        params = {"q": name, "format": "json", "addressdetails": 1, "limit": 5, "extratags": 1}
        headers = {"User-Agent": UA}
        url = "https://nominatim.openstreetmap.org/search"
        logs.append(f"OSM search: {url}?q={quote_plus(name)}")
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            logs.append(f"OSM search non-200: {r.status_code}")
            return None
        arr = r.json()
        return arr[0] if arr else None
    except Exception as e:
        logs.append(f"OSM search error: {e}")
        return None


def _nominatim_reverse(lat: float, lon: float, logs: List[str], timeout: int = 10) -> Optional[dict]:
    try:
        params = {"lat": f"{lat}", "lon": f"{lon}", "format": "json", "zoom": 16, "addressdetails": 1, "extratags": 1}
        headers = {"User-Agent": UA}
        url = "https://nominatim.openstreetmap.org/reverse"
        logs.append(f"OSM reverse: {url}?lat={lat}&lon={lon}")
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            logs.append(f"OSM reverse non-200: {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        logs.append(f"OSM reverse error: {e}")
        return None


def _extract_osm_contacts(feature: dict) -> Dict[str, str]:
    out = {"website": "", "email": "", "phone": ""}
    if not feature:
        return out
    extratags = feature.get("extratags") or {}
    out["website"] = extratags.get("website", "") or extratags.get("contact:website", "")
    out["email"] = extratags.get("email", "") or extratags.get("contact:email", "")
    out["phone"] = extratags.get("phone", "") or extratags.get("contact:phone", "")
    return out


def _duckduckgo_first_result(query: str, logs: List[str], timeout: int = 10) -> Optional[str]:
    """
    Very light HTML result scraper for first result. We avoid social sites.
    Includes polite sleep to reduce 202s.
    """
    try:
        time.sleep(1.2)  # be polite to reduce DDG 202s
        url = "https://duckduckgo.com/html/"
        params = {"q": query}
        logs.append(f"DDG search: {url}?q={quote_plus(query)}")
        r = SESSION.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            logs.append(f"DDG non-200: {r.status_code}")
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select("a.result__a"):
            href = a.get("href")
            if not href:
                continue
            full = _ensure_scheme(href)
            if not _is_social(full):
                return full
        return None
    except Exception as e:
        logs.append(f"DDG error: {e}")
        return None


def enrich_candidate_website(name: str, lat: Optional[float], lon: Optional[float], logs: List[str], country: str = "") -> Dict[str, str]:
    """
    Attempts to discover a website/email/phone for a company with the given name/coords.
    Strategy:
      1) OSM reverse by coordinates.
      2) OSM search by name (as-is and country-augmented).
      3) DuckDuckGo query "<name> <country> sitio web" then "<name>".
    """
    result = {"website": "", "email": "", "phone": ""}

    # 1) Reverse by coords
    if lat is not None and lon is not None:
        rev = _nominatim_reverse(lat, lon, logs)
        c = _extract_osm_contacts(rev or {})
        if c["website"] or c["email"] or c["phone"]:
            return c

    # 2) Name search (as-is + country)
    if name and name.strip() and name.lower() != "(unnamed)":
        feat = _nominatim_search(name, logs)
        c = _extract_osm_contacts(feat or {})
        if c["website"] or c["email"] or c["phone"]:
            return c

        if country:
            feat2 = _nominatim_search(f"{name} {country}", logs)
            c2 = _extract_osm_contacts(feat2 or {})
            if c2["website"] or c2["email"] or c2["phone"]:
                return c2

    # 3) DuckDuckGo first result
    if name:
        q1 = f"{name} {country} sitio web" if country else f"{name} sitio web"
        url = _duckduckgo_first_result(q1, logs) or _duckduckgo_first_result(name, logs)
        if url and not _is_social(url):
            result["website"] = url

    return result


# -----------------------
# Core scraping
# -----------------------
def scrape_contacts_for_candidate(
    cand: Dict[str, Any],
    timeout: int = 12,
    max_pages: int = 8,
    retries: int = 1,
    logs: List[str] = None,
    enrich_missing_website: bool = True
) -> Dict[str, Any]:
    """
    Scrape a single candidate. Returns a row of normalized fields + diagnostics.
    Optionally tries to enrich candidate website if missing (OSM + DDG).
    """
    name = cand.get("name") or "(unnamed)"
    lat = cand.get("lat")
    lon = cand.get("lon")
    dist = cand.get("distance_km")
    country_hint = cand.get("country") or ""

    base_site = _sanitize_candidate_website(cand.get("website") or "")

    row = {
        "Candidate": name,
        "Website": base_site,
        "Best Email": "",
        "Best Phone": "",
        "All Emails": "",
        "All Phones": "",
        "Pages Checked": "",
        "Lat": lat,
        "Lon": lon,
        "Distance_km": dist,
        "Source": cand.get("source"),
        "Error": "",
    }

    if logs is not None:
        logs.append(f"=== Candidate: {name} | site: {base_site or '(none)'}")

    # Enrich website if missing
    if not base_site and enrich_missing_website:
        if logs is not None:
            logs.append("No website provided; trying enrichment (OSM + DuckDuckGo)…")
        found = enrich_candidate_website(name, lat, lon, logs, country=country_hint)
        base_site = _sanitize_candidate_website(found.get("website", ""))
        # If OSM gave email/phone directly, capture now
        if not row["Best Email"] and found.get("email"):
            row["Best Email"] = found["email"]
        if not row["Best Phone"] and found.get("phone"):
            row["Best Phone"] = found["phone"]
        row["Website"] = base_site or row["Website"]

    if not base_site:
        if logs is not None:
            logs.append("No website resolved; skipping page crawl.")
        return row

    visited = []
    emails_all, phones_all = set(), set()
    domain = _domain_from_url(base_site)
    queue = [urljoin(base_site, p) for p in CANDIDATE_PATHS]
    seen = set()

    while queue and len(visited) < max_pages:
        url = _norm_url(queue.pop(0))
        if not url or url in seen:
            continue
        seen.add(url)

        resp = _get(url, timeout=timeout, retries=retries, logs=logs)
        if not resp:
            if logs is not None:
                logs.append(f"Failed to fetch {url}")
            continue

        text = resp.text or ""
        visited.append(url)

        soup = BeautifulSoup(text, "html.parser")
        emails, phones = _collect_contacts_from_html(text)
        emails_all.update(emails)
        phones_all.update(phones)

        # discover contact page
        contactish = _first_contactish_link(soup, url)
        if contactish and contactish not in seen and len(visited) + len(queue) < max_pages:
            queue.append(contactish)

    row["All Emails"] = ", ".join(sorted(emails_all))
    row["All Phones"] = ", ".join(sorted(phones_all))
    row["Pages Checked"] = ", ".join(visited)
    # Prefer domain-matching email
    best_email = _best_email(sorted(emails_all), domain)
    if best_email:
        row["Best Email"] = best_email
    if not row["Best Phone"]:
        row["Best Phone"] = (sorted(phones_all)[0] if phones_all else "")

    if logs is not None:
        logs.append(f"Collected {len(emails_all)} emails, {len(phones_all)} phones for {name}")

    return row


def scrape_contacts_bulk(
    cands: List[Dict[str, Any]],
    debug: bool = False,
    timeout: int = 12,
    max_pages: int = 8,
    retries: int = 1,
    enrich_missing_website: bool = True,
    progress_cb: Optional[Callable[[float], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Scrape a list of candidates. Returns (DataFrame, logs)
    - debug: set logger to DEBUG (also adds a header line in logs)
    - enrich_missing_website: try OSM + DuckDuckGo when no website is given
    - progress_cb: optional callback [0..1] to report progress to UI
    - log_cb: optional callback to stream the latest log lines to the UI
    """
    logs: List[str] = []
    if debug:
        logger.setLevel(logging.DEBUG)
        logs.append("Debug logging enabled.")

    if not isinstance(cands, list) or not cands:
        logs.append("No candidates passed to scraper.")
        df = pd.DataFrame([{
            "Candidate": "(none)", "Website": "", "Best Email": "",
            "Best Phone": "", "All Emails": "", "All Phones": "",
            "Pages Checked": "", "Lat": "", "Lon": "", "Distance_km": "", "Source": "", "Error": "No candidates"
        }])
        if progress_cb:
            progress_cb(1.0)
        if log_cb:
            log_cb("\n".join(logs[-50:]))
        return df, logs

    rows = []
    total = len(cands)
    for idx, c in enumerate(cands):
        try:
            msg = f"[{idx+1}/{total}] Starting candidate: {c.get('name') or '(unnamed)'}"
            logs.append(msg)
            if log_cb:
                log_cb("\n".join(logs[-50:]))

            row = scrape_contacts_for_candidate(
                c, timeout=timeout, max_pages=max_pages, retries=retries,
                logs=logs, enrich_missing_website=enrich_missing_website
            )
            rows.append(row)
        except Exception as e:
            err = str(e)
            logs.append(f"ERROR in candidate {c.get('name') or '(unnamed)'}: {err}")
            rows.append({
                "Candidate": c.get("name") or "(unnamed)", "Website": c.get("website") or "",
                "Best Email": "", "Best Phone": "", "All Emails": "", "All Phones": "",
                "Pages Checked": "", "Lat": c.get("lat"), "Lon": c.get("lon"),
                "Distance_km": c.get("distance_km"), "Source": c.get("source"),
                "Error": err
            })

        # polite delay
        time.sleep(0.25)
        if progress_cb:
            progress_cb((idx + 1) / total)
        if log_cb:
            log_cb("\n".join(logs[-50:]))

    df = pd.DataFrame(rows)

    # Sort: 1) has email 2) distance
    df["__has_email__"] = df["Best Email"].apply(lambda x: 0 if (isinstance(x, str) and "@" in x) else 1)
    sort_cols = ["__has_email__", "Distance_km", "Candidate"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last").drop(columns=["__has_email__"], errors="ignore").reset_index(index=True, drop=True)
    logs.append(f"Scraping complete. {len(df)} rows.")
    if log_cb:
        log_cb("\n".join(logs[-50:]))
    return df, logs


# Optional: backwards-compatible adapter
def scrape_contacts_bulk_compat(cands: List[Dict[str, Any]]):
    df, _ = scrape_contacts_bulk(cands, debug=False)
    return df