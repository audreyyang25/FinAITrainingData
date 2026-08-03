"""Confirm a retrieved CourtListener opinion is actually the case we asked for.

Caption search is fuzzy: "sec v. fowler" can return a different SEC v. Fowler, a
companion appeal, or an unrelated case that happens to rank. A wrong match is worse
than a miss -- it silently substitutes one document for another, and the memorization
result you compute is then about a document you never sampled.

So we compare the retrieved text against the local PDF by word-shingle containment:
what fraction of the local document's 8-grams appear in the retrieved text. Same case
scores high even when pagination, footnote placement, and headnotes differ.
"""
from __future__ import annotations
import os, re, warnings
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POOL = os.path.join(REPO, "Data Collection and Training Material Generation",
                    "raw_data", "11_Case_Law", "Core_Federal_Court_Opinions")

GOOD, PARTIAL = 0.50, 0.20


def _toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (s or "").lower())


def _shingles(toks: list[str], k: int = 8) -> set[tuple]:
    return {tuple(toks[i:i + k]) for i in range(max(0, len(toks) - k))}


def local_text(source_file: str, max_pages: int = 25) -> str:
    from pypdf import PdfReader
    p = os.path.join(POOL, source_file)
    if not os.path.exists(p):
        return ""
    try:
        r = PdfReader(p)
        return "\n".join((r.pages[i].extract_text() or "") for i in range(min(max_pages, len(r.pages))))
    except Exception:
        return ""


def containment(local: str, retrieved: str, k: int = 8) -> float:
    a, b = _shingles(_toks(local), k), _shingles(_toks(retrieved), k)
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a), 3)


def verdict(score: float) -> str:
    if score >= GOOD:
        return "match"
    if score >= PARTIAL:
        return "partial"
    return "mismatch"


def check(source_file: str, retrieved_text: str) -> dict:
    loc = local_text(source_file)
    if not loc.strip():
        return {"match_score": None, "verdict": "no_local_text"}
    s = containment(loc, retrieved_text)
    return {"match_score": s, "verdict": verdict(s),
            "local_chars": len(loc), "retrieved_chars": len(retrieved_text or "")}
