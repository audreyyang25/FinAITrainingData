"""Shared HTTP plumbing: polite rate limiting, retries, content-addressed saves, JSONL logging."""
from __future__ import annotations
import hashlib, json, os, random, threading, time, urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same convention as probe/providers.py: secrets live in the repo-root .env
# (gitignored) rather than in the shell, so a crawl works in any terminal and
# from cron without an export. Every crawl entry point imports this module, so
# loading here covers fetch_recent, fetch_courtlistener, sources and run_crawl
# at once. Real environment variables still win -- load_dotenv does not override
# what is already set, so `COURTLISTENER_TOKEN=... python ...` still works for a
# one-off.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
except ImportError:
    pass

MANIFEST_DIR = os.path.join(REPO, "Data Collection and Training Material Generation", "crawl_manifest")
RAW_DIR = os.path.join(MANIFEST_DIR, "raw")
LOG_PATH = os.path.join(MANIFEST_DIR, "crawl_log.jsonl")

# sec.gov's automated-access policy requires a declared UA with contact info.
CONTACT = os.environ.get("CRAWL_CONTACT", "").strip()
USER_AGENT = f"FinAITrainingData research crawler ({CONTACT})" if CONTACT else "FinAITrainingData research crawler"

# Per-host minimum seconds between requests. Deliberately conservative.
HOST_DELAY = {"www.sec.gov": 0.5, "www.finra.org": 1.5, "www.courtlistener.com": 3.0, "_default": 1.5}


class RateLimiter:
    def __init__(self):
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, url: str) -> None:
        host = urllib.parse.urlparse(url).netloc
        delay = HOST_DELAY.get(host, HOST_DELAY["_default"])
        with self._lock:
            prev = self._last.get(host, 0.0)
            gap = time.time() - prev
            if gap < delay:
                time.sleep(delay - gap + random.uniform(0, 0.25))
            self._last[host] = time.time()


_limiter = RateLimiter()


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    # respect_retry_after_header is deliberately off: a 429 carrying a large Retry-After
    # will otherwise park the process for however long the server asks, with no ceiling.
    # An earlier run hung for ~12h that way. Backoff is capped instead.
    retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=False)
    try:
        retry.backoff_max = 30       # urllib3 >= 2
    except Exception:
        pass
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=4))
    s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=4))
    return s


def fetch(session: requests.Session, url: str, *, timeout=(10, 60), headers: dict | None = None):
    """Return the response, or None on transport failure. Non-2xx responses are returned as-is."""
    _limiter.wait(url)
    try:
        return session.get(url, timeout=timeout, headers=headers, allow_redirects=True)
    except requests.RequestException as e:
        return e


def ext_for(resp) -> str:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "json" in ct:
        return ".json"
    if "html" in ct or "xml" in ct:
        return ".html"
    return ".bin"


def looks_like(resp, kind: str) -> bool:
    """Guard against soft-404s: a search page returned with HTTP 200 where a PDF was expected."""
    body = resp.content[:2048]
    if kind == "pdf":
        return body[:5] == b"%PDF-"
    if kind == "html":
        return b"<html" in body.lower() or b"<!doctype" in body.lower()
    return True


def save(category: str, case_id: str, resp, suffix: str = "") -> tuple[str, str, int]:
    d = os.path.join(RAW_DIR, category)
    os.makedirs(d, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in case_id)
    path = os.path.join(d, f"{safe}{suffix}{ext_for(resp)}")
    with open(path, "wb") as fh:
        fh.write(resp.content)
    return path, hashlib.sha256(resp.content).hexdigest(), len(resp.content)


def existing(category: str, case_id: str) -> str | None:
    d = os.path.join(RAW_DIR, category)
    if not os.path.isdir(d):
        return None
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in case_id)
    for f in os.listdir(d):
        stem, _, _ = f.rpartition(".")
        if stem == safe and os.path.getsize(os.path.join(d, f)) > 0:
            return os.path.join(d, f)
    return None


_log_lock = threading.Lock()


def log(rec: dict) -> None:
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with _log_lock, open(LOG_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def load_manifest() -> dict:
    with open(os.path.join(MANIFEST_DIR, "case_ids.json")) as fh:
        return json.load(fh)


def iter_cases(manifest: dict, category: str):
    """Yield (case_id, arm, record) for a category, across both arms."""
    m = manifest[category]
    if "cases" in m:
        for r in m["cases"]:
            yield r["case_id"], "carried_over", r
    else:
        for arm in ("pre_cutoff", "post_cutoff_control"):
            for r in m.get(arm, []):
                yield r["case_id"], arm, r
