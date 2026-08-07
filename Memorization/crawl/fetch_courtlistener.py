#!/usr/bin/env python3
"""Fetch the court opinions from CourtListener as plain text.

Why this category and not the others: CourtListener's `plain_text` has never been
through PDF extraction, so none of the line-wrap normalization applies -- and FreeLaw
in the Pile is CourtListener-derived, which makes this the closest thing to an actual
training-data rendering that we can obtain. The FINRA categories are robots-disallowed
and the SEC orders are PDF-only, so neither gains anything from crawling.

Output lands beside the PDF-derived datasets so all three renderings of the same 119
documents sit side by side:

  datasets/court_opinions/                PDF-derived, raw
  datasets/court_opinions_clean/          PDF-derived, wrap-collapsed
  datasets/court_opinions_courtlistener/  CourtListener plain_text   <- this script

Every retrieval is checked against the local PDF by word-shingle containment. A wrong
match is worse than a miss: it silently substitutes a different case, and the
memorization number you compute is then about a document you never sampled.

  export COURTLISTENER_TOKEN="..."
  python Memorization/crawl/fetch_courtlistener.py --probe -n 5
  python Memorization/crawl/fetch_courtlistener.py
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, urllib.parse, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Memorization.crawl import common, verify

BASE = os.path.join(common.REPO, "Data Collection and Training Material Generation")
MANIFEST = os.path.join(BASE, "crawl_manifest", "case_ids.json")
OUT_TXT = os.path.join(BASE, "datasets", "court_opinions_courtlistener")
OUT_JSON = os.path.join(BASE, "crawl_manifest", "raw", "court_opinions")
API = "https://www.courtlistener.com/api/rest/v4"

# The `cl_<year>_<court>_...` filenames use CourtListener's own court IDs, so they can
# be passed straight through as a search filter. A few need mapping.
COURT_FIX = {"calctapp": "calctapp", "delch": "delch", "dcd": "dcd"}


def headers() -> dict:
    tok = os.environ.get("COURTLISTENER_TOKEN", "").strip()
    return {"Authorization": f"Token {tok}"} if tok else {}


BUDGET = {"used": 0, "max": None}


class OutOfBudget(Exception):
    pass


def get(session, url):
    if BUDGET["max"] is not None and BUDGET["used"] >= BUDGET["max"]:
        raise OutOfBudget(f"request budget {BUDGET['max']} exhausted")
    BUDGET["used"] += 1
    r = common.fetch(session, url, headers=headers())
    if isinstance(r, Exception):
        return None, f"transport: {type(r).__name__}"
    if r.status_code == 401:
        return None, "401 unauthorized (check COURTLISTENER_TOKEN)"
    if r.status_code == 429:
        ra = r.headers.get("Retry-After", "?")
        raise OutOfBudget(f"429 daily quota exhausted; resets in {ra}s "
                          f"({int(ra)//3600}h)" if str(ra).isdigit() else "429 rate limited")
    if r.status_code != 200:
        return None, f"http {r.status_code}"
    try:
        return r.json(), None
    except ValueError:
        return None, "non-json response"


STOP = {"in", "re", "v", "vs", "the", "of", "and", "a", "an", "inc", "llc", "ltd", "co",
        "corp", "lp", "llp", "et", "al", "company", "limited"}


def query_variants(rec) -> list[str]:
    """The scrape truncated filenames at ~46 chars, so captions are often cut mid-word
    ('...limited li' for 'Litigation'). An exact-phrase search on that finds nothing.

    Since a SHA1 comparison decides identity downstream, a wrong hit costs nothing --
    only a missed hit does. So query broadly: drop the possibly-truncated tail, then
    fall back to the most distinctive tokens alone.
    """
    cap = (rec.get("caption") or rec["case_id"]).strip()
    toks = [t for t in re.split(r"[^A-Za-z0-9']+", cap) if t]
    out = [cap]
    if toks:
        # Last token is likely a truncated fragment -- try without it.
        out.append(" ".join(toks[:-1]))
    distinctive = sorted((t for t in toks if t.lower() not in STOP and len(t) > 3),
                         key=len, reverse=True)[:3]
    if distinctive:
        out.append(" ".join(distinctive))
        out.append(distinctive[0])
    seen, uniq = set(), []
    for q in out:
        q = q.strip()
        if len(q) >= 3 and q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    return uniq


def search_plan(rec) -> list[dict]:
    """Query parameter sets, most specific first."""
    court = COURT_FIX.get((rec.get("court") or "").lower(), (rec.get("court") or "").lower())
    year = rec.get("year")
    variants = query_variants(rec)
    plan = []
    if rec.get("docket"):
        p = {"q": f'docketNumber:"{rec["docket"]}"', "type": "o"}
        if court:
            p["court"] = court
        plan.append(p)
    for q in variants:
        if court:
            plan.append({"q": q, "type": "o", "court": court})
    if year and year.isdigit():
        for q in variants[:2]:
            plan.append({"q": q, "type": "o",
                         "filed_after": f"{int(year)-1}-01-01",
                         "filed_before": f"{int(year)+1}-12-31"})
    for q in variants[:2]:
        plan.append({"q": q, "type": "o"})
    seen, uniq = set(), []
    for p in plan:
        k = urllib.parse.urlencode(sorted(p.items()))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    if globals().get("DOCKET_ONLY") and rec.get("docket"):
        return uniq[:1]
    return uniq


def search(session, rec):
    """Kept for callers that just want the first non-empty result set."""
    for params in search_plan(rec):
        url = API + "/search/?" + urllib.parse.urlencode(params)
        data, err = get(session, url)
        if err:
            return None, url, err
        if data and data.get("count"):
            return data, url, None
    return None, "", "0 hits"


def opinion_ids(hit) -> list[int]:
    ids = [o["id"] for o in (hit.get("opinions") or []) if o.get("id")]
    return ids or ([hit["id"]] if hit.get("id") else [])


def best_text(op) -> tuple[str | None, str]:
    for f in ("plain_text", "html_with_citations", "html", "xml_harvard"):
        v = (op.get(f) or "").strip()
        if v:
            return f, v
    return None, ""


def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def pdf_creation_date(source_file: str) -> str | None:
    """The court's own PDF carries a CreationDate. Comparing it to the opinion's
    date_filed catches a failure that text overlap does not: retrieving the right case
    but the wrong opinion inside it (an amended opinion, a companion order)."""
    import warnings
    warnings.filterwarnings("ignore")
    p = os.path.join(verify.POOL, source_file)
    if not os.path.exists(p):
        return None
    try:
        from pypdf import PdfReader
        d = (PdfReader(p).metadata or {}).get("/CreationDate")
    except Exception:
        return None
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", str(d or ""))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def date_agreement(pdf_date: str | None, filed: str | None) -> str:
    """Same day, within a week, or clearly different."""
    if not pdf_date or not filed:
        return "unknown"
    try:
        import datetime as _dt
        a = _dt.date.fromisoformat(pdf_date)
        b = _dt.date.fromisoformat(str(filed)[:10])
    except ValueError:
        return "unknown"
    gap = abs((a - b).days)
    return "same" if gap == 0 else ("close" if gap <= 7 else f"off_by_{gap}d")


def local_sha1(source_file: str) -> str | None:
    import hashlib
    p = os.path.join(verify.POOL, source_file)
    if not os.path.exists(p):
        return None
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(session, rec):
    """Identify the right opinion, then fetch its text.

    These PDFs came from CourtListener's own storage -- the local files are byte
    identical to the copies CourtListener serves, and search results carry a sha1 per
    opinion. Identity is therefore *decidable*: matching hashes prove it is the same
    document, so a wrong hit costs nothing and only a missed hit does.

    That inverts the usual search strategy. Rather than stopping at the first query
    that returns anything, we run every query variant and scan them all for a hash
    match -- a truncated caption often returns one irrelevant hit, and stopping there
    would skip the broader query that actually finds the case. Text overlap is the
    fallback for opinions CourtListener stores without the original PDF.
    """
    want = local_sha1(rec.get("source_file", ""))
    pdf_date = pdf_creation_date(rec.get("source_file", ""))
    pool, last_url, count_seen = [], "", 0

    for params in search_plan(rec):
        last_url = API + "/search/?" + urllib.parse.urlencode(params)
        data, err = get(session, last_url)
        if err:
            return None, err, last_url
        if not data or not data.get("count"):
            continue
        count_seen += data["count"]
        for hit in data["results"][:25]:
            for o in (hit.get("opinions") or []):
                if want and (o.get("sha1") or "").lower() == want:
                    op, oerr = get(session, f"{API}/opinions/{o['id']}/")
                    if not op:
                        return None, (oerr or "opinion fetch failed"), last_url
                    field, text = best_text(op)
                    if not text:
                        return None, "hash matched but opinion has no text", last_url
                    filed = hit.get("dateFiled") or op.get("date_created")
                    return ({"opinion_id": o["id"], "field": field, "text": text,
                             "opinion": op, "cluster": hit.get("caseName"),
                             "cluster_id": hit.get("cluster_id"),
                             "search_url": last_url, "match_method": "sha1",
                             "verdict": "exact", "match_score": 1.0,
                             "date_filed": filed, "pdf_date": pdf_date,
                             "date_agreement": date_agreement(pdf_date, filed)},
                            None, last_url)
            if len(pool) < 6:
                pool.append(hit)

    # No hash match anywhere: fall back to text overlap over the hits we collected.
    best = None
    for hit in pool:
        for oid in opinion_ids(hit)[:2]:
            op, oerr = get(session, f"{API}/opinions/{oid}/")
            if not op:
                if oerr and "401" in oerr:
                    return None, oerr, last_url
                continue
            field, text = best_text(op)
            if not text:
                continue
            cmp_text = strip_tags(text) if field != "plain_text" else text
            v = verify.check(rec.get("source_file", ""), cmp_text)
            filed = hit.get("dateFiled") or op.get("date_created")
            cand = {"opinion_id": oid, "field": field, "text": text, "opinion": op,
                    "cluster": hit.get("caseName"), "cluster_id": hit.get("cluster_id"),
                    "search_url": last_url, "match_method": "text_overlap",
                    "date_filed": filed, "pdf_date": pdf_date,
                    "date_agreement": date_agreement(pdf_date, filed), **v}
            if best is None or (cand.get("match_score") or 0) > (best.get("match_score") or 0):
                best = cand
            if (cand.get("match_score") or 0) >= verify.GOOD:
                return best, None, last_url
    if best is None:
        return None, (f"{count_seen} hits, no hash match, no usable text"
                      if count_seen else "0 hits"), last_url
    return best, None, last_url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="test a few, write nothing")
    ap.add_argument("-n", type=int, default=5, help="cases to probe")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true", help="refetch even if output exists")
    ap.add_argument("--only-original-40", action="store_true")
    ap.add_argument("--max-requests", type=int, default=230,
                    help="stop before exhausting CourtListener's 250/day quota")
    ap.add_argument("--docket-only", action="store_true",
                    help="only try the docket query (2 requests/case) - cheapest mode")
    args = ap.parse_args()

    if not os.environ.get("COURTLISTENER_TOKEN"):
        print("! COURTLISTENER_TOKEN unset - anonymous access is rate-limited hard.\n"
              "  Get one at https://www.courtlistener.com/profile/api/\n")

    BUDGET["max"] = args.max_requests
    globals()["DOCKET_ONLY"] = args.docket_only
    man = json.load(open(MANIFEST))
    cases = man["court_opinions"]["cases"]
    if args.only_original_40:
        cases = [c for c in cases if c.get("in_original_40")]
    if args.probe:
        cases = cases[: args.n]
    elif args.limit:
        cases = cases[: args.limit]

    os.makedirs(OUT_TXT, exist_ok=True)
    os.makedirs(OUT_JSON, exist_ok=True)
    session = common.make_session()

    verdicts = collections.Counter()
    problems, t0 = [], time.time()
    for i, rec in enumerate(cases, 1):
        cid = rec["case_id"]
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in cid)
        dst = os.path.join(OUT_TXT, safe + ".txt")
        if not args.probe and not args.force and os.path.exists(dst) and os.path.getsize(dst) > 0:
            verdicts["skipped"] += 1
            continue

        try:
            best, err, url = resolve(session, rec)
        except OutOfBudget as e:
            print(f"\n  STOPPING: {e}")
            print(f"  {verdicts.get('exact',0)+verdicts.get('match',0)} fetched this run; "
                  f"rerun later to resume (existing output is skipped).")
            break
        if not best:
            verdicts["failed"] += 1
            problems.append((cid, err, rec.get("docket"), rec.get("caption")))
            print(f"   ! {cid[:46]:<48} {err}", flush=True)
            common.log({"phase": "courtlistener", "case_id": cid, "status": "failed",
                        "error": err, "search_url": url})
            continue

        v = best.get("verdict", "?")
        verdicts[v] += 1
        da = best.get("date_agreement", "unknown")
        if da not in ("same", "close", "unknown"):
            verdicts["date_mismatch"] += 1
        line = (f"   {cid[:44]:<46} {v:<7} via={best.get('match_method','?'):<12} "
                f"score={best.get('match_score')} date={da} {best['field']} {len(best['text'])}b")
        if v not in ("exact", "match") or da.startswith("off_by"):
            line += f"  <- got '{best.get('cluster')}'"
            problems.append((cid, f"{v}/{da}", rec.get("docket"), best.get("cluster")))
        print(line, flush=True)

        if args.probe:
            continue
        with open(dst, "w") as fh:
            fh.write(best["text"])
        with open(os.path.join(OUT_JSON, safe + ".json"), "w") as fh:
            json.dump({"manifest_record": rec, "search_url": best["search_url"],
                       "opinion_id": best["opinion_id"], "field": best["field"],
                       "match_score": best.get("match_score"), "verdict": v,
                       "match_method": best.get("match_method"),
                       "cluster_id": best.get("cluster_id"),
                       "date_filed": best.get("date_filed"), "pdf_date": best.get("pdf_date"),
                       "date_agreement": da,
                       "cluster": best.get("cluster"), "opinion": best["opinion"]}, fh, indent=1)
        common.log({"phase": "courtlistener", "case_id": cid, "status": "ok",
                    "opinion_id": best["opinion_id"], "field": best["field"],
                    "match_score": best.get("match_score"), "verdict": v,
                    "match_method": best.get("match_method"),
                    "date_agreement": da, "cluster": best.get("cluster"),
                    "chars": len(best["text"])})

    print(f"\n{'PROBE' if args.probe else 'DONE'}  {dict(verdicts)}   {time.time()-t0:.0f}s"
          f"   requests used: {BUDGET['used']}/{BUDGET['max']}")
    if problems:
        print(f"\n{len(problems)} needing review:")
        for cid, why, dk, got in problems[:25]:
            print(f"   {cid[:44]:<46} {why:<12} docket={dk} got={got}")
    if not args.probe:
        print(f"\ntext  -> {OUT_TXT}\njson  -> {OUT_JSON}")
        print("Drop anything with verdict 'mismatch' before running the experiment.")


if __name__ == "__main__":
    main()
