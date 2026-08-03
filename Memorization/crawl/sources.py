"""Per-category source resolution.

Design note: the FINRA and SEC URL patterns below are *candidates*, tried in order.
They are unverified — run `run_crawl.py --probe` first and read the report before
launching a bulk crawl. Probe output tells you which template resolves and whether
the response is a real document or a soft-404 search page.
"""
from __future__ import annotations
import os, re, json
from . import common

# --------------------------------------------------------------------- SEC
def sec_urls(case_id: str, rec: dict) -> list[str]:
    """case_id looks like 'IA-5750'; SEC orders live under the release year."""
    n = case_id.split("-", 1)[1] if "-" in case_id else case_id
    year = (rec.get("date") or "")[:4]
    years = [year] if year else []
    # An order issued in January can be filed under the prior year's directory.
    if year.isdigit():
        years.append(str(int(year) - 1))
    out = []
    for y in years:
        out += [f"https://www.sec.gov/files/litigation/admin/{y}/ia-{n}.pdf",
                f"https://www.sec.gov/litigation/admin/{y}/ia-{n}.pdf"]
    return out


# ------------------------------------------------------------------- FINRA
#
# DO NOT CRAWL. finra.org/robots.txt disallows exactly these paths:
#     Disallow: /sites/default/files/aao_documents/*    (arbitration awards)
#     Disallow: /sites/default/files/fda_documents/*    (disciplinary actions / AWCs)
#     Disallow: /sites/default/files/non-indexed/*
#
# Two consequences:
#   1. Fetching them would ignore a stated crawl policy, and we already hold every one
#      of these PDFs in raw_data -- there is nothing to gain.
#   2. More importantly for the study: Common Crawl's CCBot honors robots.txt, so these
#      documents are absent from Common Crawl and therefore from any CC-derived
#      pretraining corpus. That makes the FINRA categories structural negatives, which
#      is useful -- any "extraction" reported on them is measuring format, not content.
#
# Widely-quoted excerpts may still appear via third-party reposts, so fragment-level
# memorization is not ruled out. Full-document extraction is.

FINRA_ROBOTS_DISALLOWED = True


def finra_blocked(case_id: str, rec: dict) -> list[str]:
    raise RuntimeError(
        "FINRA document paths are robots-disallowed and already present in raw_data. "
        "Use the local PDFs with Memorization/extract/normalize.py instead."
    )


# ----------------------------------------------------------- CourtListener
CL_API = "https://www.courtlistener.com/api/rest/v4"


def cl_headers() -> dict:
    tok = os.environ.get("COURTLISTENER_TOKEN", "").strip()
    return {"Authorization": f"Token {tok}"} if tok else {}


def cl_search(session, rec: dict):
    """Search by docket number when we have one, else by caption. Returns parsed JSON."""
    q = rec.get("docket") or rec.get("caption") or rec["case_id"]
    params = {"q": q, "type": "o"}
    if rec.get("year"):
        params["filed_after"] = f"{int(rec['year'])-1}-01-01"
        params["filed_before"] = f"{int(rec['year'])+1}-12-31"
    url = CL_API + "/search/?" + "&".join(f"{k}={requests_quote(str(v))}" for k, v in params.items())
    r = common.fetch(session, url, headers=cl_headers())
    if isinstance(r, Exception) or r.status_code != 200:
        return None, url, r
    try:
        return r.json(), url, r
    except ValueError:
        return None, url, r


def cl_opinion(session, opinion_id: int):
    url = f"{CL_API}/opinions/{opinion_id}/"
    r = common.fetch(session, url, headers=cl_headers())
    if isinstance(r, Exception) or r.status_code != 200:
        return None, url, r
    try:
        return r.json(), url, r
    except ValueError:
        return None, url, r


def requests_quote(s: str) -> str:
    import urllib.parse
    return urllib.parse.quote_plus(s)


# --------------------------------------------------------------------- map
# kind: what we expect back, used to detect soft-404s.
SOURCES = {
    "court_opinions":     {"urls": None,           "kind": "json", "host": "courtlistener.com",
                           "crawl": True,
                           "note": "plain_text field; FreeLaw in the Pile is CourtListener-derived"},
    "sec_ia_releases":    {"urls": sec_urls,       "kind": "pdf",  "host": "sec.gov",
                           "crawl": True,
                           "note": "path is crawlable (only /search/ is disallowed), but PDF-only "
                                   "-- identical to raw_data. Crawl only to confirm availability."},
    "arbitration_awards": {"urls": finra_blocked,  "kind": "pdf",  "host": "finra.org",
                           "crawl": False, "note": "robots-disallowed; use local PDFs"},
    "finra_awc":          {"urls": finra_blocked,  "kind": "pdf",  "host": "finra.org",
                           "crawl": False, "note": "robots-disallowed; use local PDFs"},
    "oho_decisions":      {"urls": finra_blocked,  "kind": "pdf",  "host": "finra.org",
                           "crawl": False, "note": "robots-disallowed; use local PDFs"},
}

CRAWLABLE = [k for k, v in SOURCES.items() if v.get("crawl")]
