#!/usr/bin/env python3
"""Expand the manifest's court_opinions entry to every opinion in the local pool.

The manifest carried 40 sampled opinions. For a full run we want all of them, with
whatever identifiers CourtListener can be searched on: docket number where the opinion
states one, otherwise the caption parsed from the filename.

Writes back into crawl_manifest/case_ids.json. Records `in_original_40` so the earlier
sample stays recoverable.

  python Memorization/crawl/expand_court.py
"""
from __future__ import annotations
import json, os, re, sys, warnings, glob
warnings.filterwarnings("ignore")
from pypdf import PdfReader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
POOL = os.path.join(BASE, "raw_data", "11_Case_Law", "Core_Federal_Court_Opinions")
MANIFEST = os.path.join(BASE, "crawl_manifest", "case_ids.json")

DOCKET_PATS = [
    r"\bNos?\.\s*([0-9]{1,2}[-\u2013][0-9]{3,5}(?:\s*,\s*[0-9]{1,2}[-\u2013][0-9]{3,5})*)",
    r"\b(?:Civil|Criminal)\s+(?:Action\s+)?No\.\s*([0-9A-Za-z:\-]+)",
    r"\bCase\s+No\.\s*([0-9A-Za-z:\-]+)",
    r"\bC\.?A\.?\s+No\.\s*([0-9A-Za-z:\-]+)",
    r"\bNo\.\s*([0-9]{2}[-\u2013][0-9]{3,5})",
]
COURT_NAME = {
    "ca1": "First Circuit", "ca2": "Second Circuit", "ca3": "Third Circuit",
    "ca4": "Fourth Circuit", "ca5": "Fifth Circuit", "ca6": "Sixth Circuit",
    "ca7": "Seventh Circuit", "ca8": "Eighth Circuit", "ca9": "Ninth Circuit",
    "ca10": "Tenth Circuit", "ca11": "Eleventh Circuit", "cadc": "D.C. Circuit",
    "cafc": "Federal Circuit", "scotus": "Supreme Court", "dcd": "D.D.C.",
    "delch": "Delaware Chancery",
}


def head_text(path, pages=2):
    try:
        r = PdfReader(path)
        return "\n".join((r.pages[i].extract_text() or "") for i in range(min(pages, len(r.pages)))), len(r.pages)
    except Exception:
        return "", -1


def main():
    man = json.load(open(MANIFEST))
    prior = {r["case_id"] for r in man.get("court_opinions", {}).get("cases", [])}
    files = sorted(glob.glob(os.path.join(POOL, "*.pdf")) + glob.glob(os.path.join(POOL, "*.PDF")))
    out = []
    for i, p in enumerate(files, 1):
        base = os.path.basename(p)
        m = re.match(r"^cl_(\d{4})_([a-z0-9]+)_(.+)\.pdf$", base, re.I)
        year, court, cap = (m.group(1), m.group(2), m.group(3)) if m else ("", "", base[:-4])
        caption = cap.replace("_", " ").replace(".", ". ").strip()
        caption = re.sub(r"\s+", " ", caption)
        txt, npages = head_text(p)
        docket = None
        for pat in DOCKET_PATS:
            mm = re.search(pat, txt)
            if mm:
                docket = re.sub(r"\s+", " ", mm.group(1)).strip()
                break
        out.append({
            "case_id": base[:-4],
            "year": year or None,
            "court": court or None,
            "court_name": COURT_NAME.get(court.lower(), court) if court else None,
            "caption": caption,
            "docket": docket,
            "n_pages": npages,
            "source_file": base,
            "in_original_40": base[:-4] in prior,
            "url_hint": "https://www.courtlistener.com/?q=" + re.sub(r"\s+", "+", caption),
        })
        if i % 25 == 0:
            print(f"   {i}/{len(files)}", flush=True)

    man["court_opinions"] = {
        "cutoff": None,
        "note": "full local pool; superset of the original 40 (see in_original_40)",
        "cases": out,
    }
    json.dump(man, open(MANIFEST, "w"), indent=1)

    with_docket = sum(1 for r in out if r["docket"])
    print(f"\ncourt_opinions: {len(out)} cases  ({with_docket} with docket, "
          f"{len(out)-with_docket} caption-only)")
    print(f"  carried from original 40: {sum(1 for r in out if r['in_original_40'])}")
    from collections import Counter
    print("  by court:", dict(Counter(r["court"] for r in out).most_common(10)))
    print("  by year :", dict(sorted(Counter(r["year"] for r in out).items())))
    print(f"\nwrote {MANIFEST}")


if __name__ == "__main__":
    main()
