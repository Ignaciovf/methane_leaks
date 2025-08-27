import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import contact_scraper as cs


def test_collect_contacts_from_html_basic():
    html = """
    <div>email us at env@acme-energy.com or press@acme-energy.com</div>
    <div>Call +1 (713) 555-1234</div>
    """
    emails, phones = cs._collect_contacts_from_html(html)
    assert "env@acme-energy.com" in emails
    assert "press@acme-energy.com" in emails
    assert any("713" in p for p in phones)


def test_best_email_prefers_domain():
    emails = ["contact@gmail.com", "ir@acme.com", "noreply@acme.com"]
    chosen = cs._best_email(emails, "acme.com")
    assert chosen == "ir@acme.com"


def test_best_email_role_fallback():
    emails = ["info@vendor.net", "press@vendor.net"]
    chosen = cs._best_email(emails, "acme.com")
    assert chosen == "press@vendor.net"


def test_domain_from_url():
    assert cs._domain_from_url("https://www.example.co.uk/x") == "example.co.uk"
    assert cs._domain_from_url("http://acme.com") == "acme.com"


def test_resolve_website_uses_given(monkeypatch):
    logs = []
    site = cs.resolve_website(
        name="Five Sisters compressor station",
        country="USA",
        lat=None,
        lon=None,
        given_website="https://operator.example",
        logs=logs,
    )
    assert site.startswith("https://operator.example")


def test_resolve_website_uses_overrides(tmp_path, monkeypatch):
    ov = tmp_path / "site_overrides.json"
    ov.write_text('{"five sisters": "https://operator.example"}', encoding="utf-8")
    monkeypatch.setenv("SCRAPER_SITE_OVERRIDES", str(ov))

    logs = []
    site = cs.resolve_website(
        name="Five Sisters compressor station",
        country="USA",
        lat=None,
        lon=None,
        given_website="",
        logs=logs,
    )
    assert site == "https://operator.example"


def test_resolve_website_uses_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    def fake_openai(query, logs, timeout=10):
        return "https://operator.example"

    monkeypatch.setattr(cs, "_openai_web_search_first_result", fake_openai)

    logs = []
    site = cs.resolve_website(
        name="Five Sisters compressor station",
        country="USA",
        lat=None,
        lon=None,
        given_website="",
        logs=logs,
    )
    assert site == "https://operator.example"

