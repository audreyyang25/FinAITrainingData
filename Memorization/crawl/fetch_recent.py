#!/usr/bin/env python3
"""Expand the post-cutoff control arm with recent securities opinions.

The control arm is the scarce resource: only 12 cases postdate 2024-12-31, and
2025H1 is empty entirely. Unlike the FINRA categories (robots-disallowed, fixed
pool), CourtListener carries every opinion, so this arm is cheap to grow.

Efficiency matters because of the 250 requests/day quota. Searching by date range
returns many candidates per request, so the cost is ~1 search per page plus 1
fetch per opinion -- roughly 200 new cases per day, versus the ~2 requests/case
the docket-lookup path needed.

New cases are appended to the manifest and written to the CourtListener text
directory, so build_questions.py picks them up with no changes.

  python Memorization/crawl/fetch_recent.py --dry-run
  python Memorization/crawl/fetch_recent.py --after 2024-07-01 --max-requests 230
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys, time, urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Memorization.crawl import common

BASE = os.path.join(common.REPO, "Data Collection and Training Material Generation")
MANIFEST = os.path.join(BASE, "crawl_manifest", "case_ids.json")
TXT_DIR = os.path.join(BASE, "datasets", "court_opinions_courtlistener")
JSON_DIR = os.path.join(BASE, "crawl_manifest", "raw", "court_opinions")
API = "https://www.courtlistener.com/api/rest/v4"

# Match the existing corpus's domain: securities / investment-adviser / broker
# litigation. Broad enough to fill an arm, narrow enough to stay comparable to
# the pre-cutoff cases rather than introducing a subject-matter confound.
QUERIES = [
    '"securities fraud"', '"Securities Exchange Act"', '"investment adviser"',
    '"Securities and Exchange Commission"', '"broker-dealer"',
    '"Rule 10b-5"', '"breach of fiduciary duty" securities',
]
# Courts weighted toward the existing distribution (ca2 37, ca9 20, dcd 16, ...).
COURTS = ["ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9",
          "ca10", "ca11", "cadc", "scotus", "dcd", "nysd", "cand", "delch"]

CIRCUIT = re.compile(r"^(ca\d{1,2}|cadc|cafc)$")
DISTRICT = re.compile(r"^[a-z]{2}[ncsewmd]?d$")


def classify(court: str):
    c = (court or "").lower()
    if c == "scotus":
        return "federal", "supreme"
    if CIRCUIT.match(c):
        return "federal", "circuit"
    if DISTRICT.match(c) or c in {"dcd", "nysd", "cand"}:
        return "federal", "district"
    return "state", "state"


def headers():
    tok = os.environ.get("COURTLISTENER_TOKEN", "").strip()
    return {"Authorization": f"Token {tok}"} if tok else {}


class Budget:
    def __init__(self, cap): self.cap, self.used = cap, 0
    def spend(self):
        if self.used >= self.cap:
            raise RuntimeError(f"request budget {self.cap} exhausted")
        self.used += 1


def get(session, url, budget):
    budget.spend()
    r = common.fetch(session, url, headers=headers())
    if isinstance(r, Exception):
        # The session's retry adapter has 429 in its status_forcelist, so a
        # throttled request is retried three times and surfaces as RetryError
        # rather than a 429 we can read. Treat it as throttling and stop --
        # otherwise the run silently burns its whole budget on 429s.
        name = type(r).__name__
        if "Retry" in name or "TooManyRedirects" in name:
            raise RuntimeError(f"throttled (RetryError) after {budget.used} requests — "
                               f"CourtListener allows 100/hour; wait and resume")
        return None, f"transport: {name}"
    if r.status_code == 429:
        ra = r.headers.get("Retry-After", "?")
        raise RuntimeError(f"429 throttled after {budget.used} requests; "
                           f"resets in {ra}s")
    if r.status_code != 200:
        return None, f"http {r.status_code}"
    try:
        return r.json(), None
    except ValueError:
        return None, "non-json"


def slugify(caption: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (caption or "").lower()).strip("_")
    return re.sub(r"_+", "_", s)[:46]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", default="2024-07-01", help="filed_after date")
    ap.add_argument("--before", default="", help="optional filed_before date")
    ap.add_argument("--max-requests", type=int, default=90,
                    help="CourtListener throttles at 100/hour (and 250/day). The hourly "
                         "cap is the binding one for a burst — stay under it.")
    ap.add_argument("--per-window", type=int, default=6,
                    help="fetch round-robin across date windows so the budget spreads "
                         "across the range instead of piling into the newest weeks")
    ap.add_argument("--min-chars", type=int, default=4000,
                    help="skip opinions too short to yield questions")
    ap.add_argument("--dry-run", action="store_true", help="search only, fetch nothing")
    args = ap.parse_args()

    if not os.environ.get("COURTLISTENER_TOKEN"):
        print("! COURTLISTENER_TOKEN unset — the opinions endpoint requires auth\n")

    man = json.load(open(MANIFEST))
    cases = man["court_opinions"]["cases"]
    have_ids = {c["case_id"] for c in cases}
    have_opinion = {c.get("opinion_id") for c in cases if c.get("opinion_id")}
    print(f"manifest has {len(cases)} court opinions; searching for more filed after {args.after}\n")

    session = common.make_session()
    budget = Budget(args.max_requests)
    os.makedirs(TXT_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

    # ---- phase 1: discover candidates (cheap — many per request) -------------
    # Stratify by date window. Sorting by dateFiled desc alone fills every page
    # with the newest term and leaves earlier windows (notably the empty 2025H1)
    # unrepresented -- which would cost the dose-response band between model
    # cutoffs, where a case is post-cutoff for one model and pre-cutoff for another.
    def windows(start: str, end: str) -> list[tuple[str, str]]:
        out, y = [], int(start[:4])
        while y <= int((end or "2026-12-31")[:4]):
            for a, b in ((f"{y}-01-01", f"{y}-06-30"), (f"{y}-07-01", f"{y}-12-31")):
                if b >= start and (not end or a <= end):
                    out.append((max(a, start), min(b, end) if end else b))
            y += 1
        return out

    seen, candidates = set(), []
    plan = [(q, w) for w in windows(args.after, args.before) for q in QUERIES]
    try:
        for q, (wa, wb) in plan:
            params = {"q": q, "type": "o", "order_by": "dateFiled desc",
                      "filed_after": wa, "filed_before": wb}
            url = API + "/search/?" + urllib.parse.urlencode(params)
            data, err = get(session, url, budget)
            if err or not data:
                continue
            n = 0
            for hit in data.get("results", []):
                for o in (hit.get("opinions") or []):
                    oid = o.get("id")
                    if not oid or oid in seen or oid in have_opinion:
                        continue
                    seen.add(oid)
                    candidates.append({
                        "opinion_id": oid, "cluster_id": hit.get("cluster_id"),
                        "caption": hit.get("caseName") or "", "court": hit.get("court_id") or "",
                        "date_filed": (hit.get("dateFiled") or "")[:10],
                        "docket": hit.get("docketNumber") or None,
                    })
                    n += 1
            if n:
                print(f"   {wa[:7]}..{wb[:7]}  {q[:32]:<34} +{n:>3}   ({budget.used}/{budget.cap})")
    except RuntimeError as e:
        print(f"\n  stopped during search: {e}")

    candidates = [c for c in candidates if c["date_filed"] > args.after]
    # Round-robin across half-year buckets. Plain date-desc ordering spent the
    # entire budget inside a three-week window last run and left 2025H1 empty
    # even though 60 candidates had been found for it.
    buckets = collections.OrderedDict()
    for c in sorted(candidates, key=lambda c: c["date_filed"], reverse=True):
        key = c["date_filed"][:4] + ("H1" if c["date_filed"][5:7] <= "06" else "H2")
        buckets.setdefault(key, []).append(c)
    interleaved = []
    while any(buckets.values()):
        for k in list(buckets):
            take = buckets[k][: args.per_window]
            del buckets[k][: args.per_window]
            interleaved.extend(take)
            if not buckets[k]:
                del buckets[k]
    candidates = interleaved
    by_half = collections.Counter(c["date_filed"][:4] + ("H1" if c["date_filed"][5:7] <= "06" else "H2")
                                  for c in candidates)
    print(f"\n{len(candidates)} unique new candidates: {dict(sorted(by_half.items()))}")

    if args.dry_run:
        for c in candidates[:12]:
            print(f"   {c['date_filed']}  {c['court']:<8} {c['caption'][:58]}")
        print(f"\n(dry run — would fetch up to {budget.cap - budget.used} of them)")
        return

    # ---- phase 2: fetch text (1 request each) --------------------------------
    added, skipped_short, failed = [], 0, 0
    try:
        for c in candidates:
            op, err = get(session, f"{API}/opinions/{c['opinion_id']}/", budget)
            if not op:
                failed += 1
                continue
            text = ""
            field = ""
            for f in ("plain_text", "html_with_citations", "html"):
                if (op.get(f) or "").strip():
                    field, text = f, op[f]
                    break
            if field != "plain_text":
                text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
            if len(text) < args.min_chars:
                skipped_short += 1
                continue

            yr = c["date_filed"][:4]
            cid = f"cl_{yr}_{c['court']}_{slugify(c['caption'])}"
            if cid in have_ids:
                cid = f"{cid}_{c['opinion_id']}"
            have_ids.add(cid)
            juris, level = classify(c["court"])

            open(os.path.join(TXT_DIR, cid + ".txt"), "w").write(text)
            json.dump({"manifest_record": {"case_id": cid}, "opinion_id": c["opinion_id"],
                       "field": field, "verdict": "fetched_by_id",
                       "date_filed": c["date_filed"], "cluster": c["caption"],
                       "opinion": op},
                      open(os.path.join(JSON_DIR, cid + ".json"), "w"), indent=1)

            rec = {"case_id": cid, "year": yr, "court": c["court"], "court_name": c["court"],
                   "caption": c["caption"], "docket": c["docket"], "n_pages": None,
                   "source_file": None, "in_original_40": False,
                   "url_hint": f"https://www.courtlistener.com/opinion/{c['cluster_id']}/",
                   "jurisdiction": juris, "court_level": level,
                   "arm": "post_cutoff_control", "date_filed": c["date_filed"],
                   "date_source": "courtlistener", "opinion_id": c["opinion_id"],
                   "cluster_id": c["cluster_id"], "cl_verdict": "fetched_by_id",
                   "cl_field": field, "added_by": "fetch_recent"}
            cases.append(rec)
            added.append(rec)
            if len(added) % 10 == 0:
                print(f"   fetched {len(added)}  (requests {budget.used}/{budget.cap})", flush=True)
    except RuntimeError as e:
        print(f"\n  stopped: {e}")

    man["court_opinions"]["cases"] = cases
    json.dump(man, open(MANIFEST, "w"), indent=1)

    print(f"\nadded {len(added)} opinions "
          f"(skipped {skipped_short} too short, {failed} fetch failures)")
    print(f"requests used: {budget.used}/{budget.cap}")
    if added:
        h = collections.Counter(r["date_filed"][:4] + ("H1" if r["date_filed"][5:7] <= "06" else "H2")
                                for r in added)
        print(f"new by half-year: {dict(sorted(h.items()))}")
        print(f"manifest now has {len(cases)} court opinions")
        print("\nnext: python Memorization/qa/build_questions.py")


if __name__ == "__main__":
    main()
