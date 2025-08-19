import os
import re
import time
import html
import math
import json
import tldextract
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# Optional OpenAI (for choosing best contact when multiple)
try:
    from openai import OpenAI
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    _OPENAI = OpenAI() if _OPENAI_KEY else None
    _EMBED_MODEL = "text-embedding-3-small"
    _CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
except Exception:
    _OPENAI = None

UA = "methane-leak-notifier/1.0 (contact-scraper)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"})

EMAIL_RE = re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.I)
PHONE_RE = re.compile(r'(\+?\d[\d\s().-]{6,}\d)')

CANDIDATE_PATHS = ["/", "/contact", "/contact-us", "/about", "/about-us", "/imprint", "/legal", "/empresa", "/contacto", "/quienes-somos"]

def _norm_url(url: str) -> str:
    return url.split("#")[0].strip()

def _ensure_scheme(url: str) -> str:
    if not url:
        return url
    if not urlparse(url).scheme:
        return "https://" + url
    return url

def _fetch(url: str, timeout=20) -> Optional[str]:
    try:
        r = SESSION.get(url, timeout=timeout)
        if r.status_code >= 400:
            return None
        ct = (r.headers.get("Content-Type") or "").lower()
        if "text/html" not in ct and "application/xhtml" not in ct:
            return None
        return r.text
    except Exception:
        return None

def _extract_contacts(html_text: str) -> Tuple[List[str], List[str]]:
    emails = set(e.lower() for e in EMAIL_RE.findall(html_text or ""))
    phones = set(PHONE_RE.findall(html_text or ""))
    return sorted(emails), sorted(phones)

def _crawl(domain_or_url: str) -> Dict[str, Any]:
    base = _ensure_scheme(domain_or_url)
    # If given a full URL, reduce to scheme + netloc
    p = urlparse(base)
    base_root = f"{p.scheme}://{p.netloc}" if p.netloc else base
    found_emails, found_phones = set(), set()
    visited = set()
    pages_checked = []
    for path in CANDIDATE_PATHS:
        url = urljoin(base_root, path)
        url = _norm_url(url)
        if url in visited: continue
        visited.add(url)
        html_text = _fetch(url)
        if not html_text: continue
        pages_checked.append(url)
        emails, phones = _extract_contacts(html_text)
        found_emails.update(emails); found_phones.update(phones)
        # also parse some obvious mailto/tel links
        soup = BeautifulSoup(html_text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("mailto:"):
                found_emails.add(href.split("mailto:")[-1].split("?")[0].lower())
            if href.startswith("tel:"):
                found_phones.add(href.split("tel:")[-1])
    return {
        "pages_checked": pages_checked,
        "emails": sorted(found_emails),
        "phones": sorted(found_phones),
    }

def _score_email_heuristic(email: str, company_name: str) -> int:
    # Prefer env/compliance/sustainability/reporting/contact addresses
    s = email.lower()
    score = 0
    hits = ["environment", "medioambiente", "sustainab", "ambiental", "compliance",
            "legal", "info", "contact", "support", "press", "report", "denuncia"]
    for h in hits:
        if h in s: score += 1
    # Company domain match
    if company_name and tldextract.extract(s.split("@")[-1]).domain in company_name.lower().replace(" ", ""):
        score += 1
    return score

def _choose_best_contact(company_name: str, emails: List[str], phones: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if not emails and not phones:
        return None, None
    if _OPENAI and emails:
        # Ask model to pick best email for environmental incident notifications
        prompt = f"""
You are selecting the best contact email for reporting an urgent methane leak to a company.
Company: {company_name}
Emails: {emails}

Pick a single address that is most likely monitored by the company for environmental/legal notifications (e.g., environment@, compliance@, info@ if nothing else). Respond with ONLY the email address.
"""
        try:
            resp = _OPENAI.chat.completions.create(
                model=_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            choice = (resp.choices[0].message.content or "").strip()
            if EMAIL_RE.fullmatch(choice):
                best_email = choice
            else:
                # fallback to heuristic
                best_email = max(emails, key=lambda e: _score_email_heuristic(e, company_name))
        except Exception:
            best_email = max(emails, key=lambda e: _score_email_heuristic(e, company_name))
    elif emails:
        best_email = max(emails, key=lambda e: _score_email_heuristic(e, company_name))
    else:
        best_email = None

    best_phone = phones[0] if phones else None
    return best_email, best_phone

def scrape_contacts_for_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
    name = cand.get("name") or "(unnamed)"
    website = cand.get("website")
    # Try to infer domain from raw/context if no website
    if not website:
        # Some Mapbox features include URL-like strings in raw props; try common keys
        raw = cand.get("raw", {}) or {}
        props = raw.get("properties", {})
        for k in ("website", "url", "contact:website"):
            if props.get(k):
                website = props.get(k)
                break

    emails, phones, pages = [], [], []
    if website:
        crawled = _crawl(website)
        emails = crawled["emails"]; phones = crawled["phones"]; pages = crawled["pages_checked"]

    best_email, best_phone = _choose_best_contact(name, emails, phones)

    return {
        "Candidate": name,
        "Website": website or "",
        "Best Email": best_email or "",
        "Best Phone": best_phone or "",
        "All Emails": ", ".join(emails[:10]),
        "All Phones": ", ".join(phones[:10]),
        "Pages Checked": ", ".join(pages[:5]),
        "Lat": cand.get("lat"), "Lon": cand.get("lon"),
        "Distance_km": cand.get("distance_km"),
        "Source": cand.get("source"),
    }

def scrape_contacts_bulk(cands: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for c in cands:
        try:
            rows.append(scrape_contacts_for_candidate(c))
            time.sleep(0.3)  # be polite
        except Exception as e:
            rows.append({"Candidate": c.get("name") or "(unnamed)", "Website": "", "Best Email": "",
                         "Best Phone": "", "All Emails": "", "All Phones": "", "Pages Checked": "",
                         "Lat": c.get("lat"), "Lon": c.get("lon"), "Distance_km": c.get("distance_km"),
                         "Source": c.get("source"), "Error": str(e)})
    df = pd.DataFrame(rows)
    # Sort: have email first, then by distance
    has_email = df["Best Email"].apply(lambda x: 0 if (x and "@" in x) else 1)
    df = df.sort_values(["Candidate", "Distance_km", has_email], na_position="last").reset_index(drop=True)
    return df