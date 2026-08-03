#!/usr/bin/env python3
"""Crawl driver.

Run --probe FIRST. It tests a few IDs per category and reports what each source
actually returns, so you can decide whether a bulk crawl is worth doing at all.
Any category that comes back PDF-only duplicates what is already in raw_data --
skip it and extract locally instead (see Memorization/extract/compare_extractors.py).

  export CRAWL_CONTACT="you@example.com"        # required by sec.gov policy
  export COURTLISTENER_TOKEN="..."              # https://www.courtlistener.com/profile/api/

  python -m Memorization.crawl.run_crawl --probe
  python -m Memorization.crawl.run_crawl --category court_opinions
"""
from __future__ import annotations
import argparse, json, os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Memorization.crawl import common, sources, verify


def probe_one(session, category, case_id, rec):
    spec = sources.SOURCES[category]
    if category == "court_opinions":
        data, url, r = sources.cl_search(session, rec)
        if data is None:
            code = getattr(r, "status_code", type(r).__name__)
            return {"case_id": case_id, "url": url, "result": f"search failed ({code})"}
        n = data.get("count", 0)
        if not n:
            return {"case_id": case_id, "url": url, "result": "0 hits"}
        top = data["results"][0]
        oid = (top.get("opinions") or [{}])[0].get("id") or top.get("id")
        op, ourl, orr = sources.cl_opinion(session, oid) if oid else (None, "", None)
        fields = [k for k in ("plain_text", "html", "html_with_citations", "xml_harvard")
                  if op and (op.get(k) or "").strip()]
        sizes = {k: len(op[k]) for k in fields} if op else {}
        return {"case_id": case_id, "url": url, "result": f"{n} hits", "opinion_id": oid,
                "text_fields": sizes, "cluster": top.get("caseName")}

    for url in spec["urls"](case_id, rec):
        r = common.fetch(session, url)
        if isinstance(r, Exception):
            continue
        if r.status_code == 200 and common.looks_like(r, spec["kind"]):
            ct = (r.headers.get("Content-Type") or "").split(";")[0]
            return {"case_id": case_id, "url": url, "result": f"200 {ct}", "bytes": len(r.content)}
        if r.status_code == 200:
            ct = (r.headers.get("Content-Type") or "").split(";")[0]
            return {"case_id": case_id, "url": url,
                    "result": f"200 but not {spec['kind']} (soft-404?) ct={ct}", "bytes": len(r.content)}
    return {"case_id": case_id, "url": url, "result": "all candidates failed"}


def cmd_probe(args, manifest, session):
    cats = args.category or sources.CRAWLABLE
    print(f"UA: {common.USER_AGENT}")
    if not common.CONTACT:
        print("  ! CRAWL_CONTACT unset - sec.gov asks for a contact address in the UA")
    if not os.environ.get("COURTLISTENER_TOKEN"):
        print("  ! COURTLISTENER_TOKEN unset - CourtListener rate-limits anonymous access hard")
    for cat in cats:
        if not sources.SOURCES[cat].get("crawl"):
            print(f"\n=== {cat}: SKIPPED - {sources.SOURCES[cat]['note']}")
            continue
        cases = list(common.iter_cases(manifest, cat))[: args.n]
        print(f"\n=== {cat}  (probing {len(cases)})")
        for cid, arm, rec in cases:
            out = probe_one(session, cat, cid, rec)
            print(f"   {cid:<34} {out['result']}"
                  + (f"  fields={out['text_fields']}" if out.get("text_fields") else "")
                  + (f"  {out.get('bytes','')}b" if out.get("bytes") else ""))
            common.log({"phase": "probe", "category": cat, **out})
    print("\nIf a category returns PDF only, do not bulk-crawl it -- you already have those "
          "PDFs in raw_data. Extract locally instead.")


def cmd_crawl(args, manifest, session):
    cats = args.category or sources.CRAWLABLE
    for cat in cats:
        if not sources.SOURCES[cat].get("crawl"):
            print(f"\n=== {cat}: SKIPPED - {sources.SOURCES[cat]['note']}")
            continue
        cases = list(common.iter_cases(manifest, cat))
        if args.limit:
            cases = cases[: args.limit]
        got = skip = fail = 0
        print(f"\n=== {cat}: {len(cases)} cases")
        for i, (cid, arm, rec) in enumerate(cases, 1):
            if not args.force and common.existing(cat, cid):
                skip += 1
                continue
            if args.dry_run:
                urls = ["<courtlistener search>"] if cat == "court_opinions" else sources.SOURCES[cat]["urls"](cid, rec)
                print(f"   {cid} -> {urls[0]}")
                continue

            if cat == "court_opinions":
                data, url, r = sources.cl_search(session, rec)
                oid = None
                if data and data.get("count"):
                    top = data["results"][0]
                    oid = (top.get("opinions") or [{}])[0].get("id") or top.get("id")
                if not oid:
                    fail += 1
                    common.log({"phase": "crawl", "category": cat, "arm": arm, "case_id": cid,
                                "url": url, "status": "no_match"})
                    continue
                op, ourl, orr = sources.cl_opinion(session, oid)
                if not op:
                    fail += 1
                    common.log({"phase": "crawl", "category": cat, "arm": arm, "case_id": cid,
                                "url": ourl, "status": "opinion_fetch_failed"})
                    continue
                d = os.path.join(common.RAW_DIR, cat)
                os.makedirs(d, exist_ok=True)
                safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in cid)
                with open(os.path.join(d, safe + ".json"), "w") as fh:
                    json.dump({"manifest_record": rec, "search_url": url,
                               "opinion_url": ourl, "opinion": op}, fh, indent=1)
                chosen, text = None, ""
                for field in ("plain_text", "html_with_citations", "html"):
                    if (op.get(field) or "").strip():
                        chosen, text = field, op[field]
                        with open(os.path.join(d, f"{safe}.{field}.txt"), "w") as fh:
                            fh.write(text)
                        break
                v = verify.check(rec.get("source_file", ""), text) if chosen else {"verdict": "no_text"}
                got += 1
                if v.get("verdict") in ("mismatch", "no_text"):
                    print(f"   ! {cid}: {v.get('verdict')} (score={v.get('match_score')})", flush=True)
                common.log({"phase": "crawl", "category": cat, "arm": arm, "case_id": cid,
                            "url": ourl, "status": "ok", "opinion_id": oid, "field": chosen,
                            "caption_query": rec.get("docket") or rec.get("caption"),
                            **v,
                            "fields": {k: len(op.get(k) or "") for k in
                                       ("plain_text", "html", "html_with_citations")}})
                continue

            spec = sources.SOURCES[cat]
            done = False
            for url in spec["urls"](cid, rec):
                r = common.fetch(session, url)
                if isinstance(r, Exception):
                    continue
                if r.status_code == 200 and common.looks_like(r, spec["kind"]):
                    path, sha, n = common.save(cat, cid, r)
                    got += 1
                    done = True
                    common.log({"phase": "crawl", "category": cat, "arm": arm, "case_id": cid,
                                "url": url, "status": "ok", "path": os.path.relpath(path, common.MANIFEST_DIR),
                                "sha256": sha, "bytes": n})
                    break
            if not done:
                fail += 1
                common.log({"phase": "crawl", "category": cat, "arm": arm, "case_id": cid,
                            "status": "failed"})
            if i % 20 == 0:
                print(f"   {i}/{len(cases)}  ok={got} skip={skip} fail={fail}", flush=True)
        print(f"   done: ok={got} skipped={skip} failed={fail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="test a few IDs per category, fetch nothing in bulk")
    ap.add_argument("--category", action="append", choices=list(sources.SOURCES))
    ap.add_argument("-n", type=int, default=3, help="cases per category when probing")
    ap.add_argument("--limit", type=int, help="cap cases per category when crawling")
    ap.add_argument("--dry-run", action="store_true", help="print URLs, fetch nothing")
    ap.add_argument("--force", action="store_true", help="refetch even if output exists")
    args = ap.parse_args()

    manifest = common.load_manifest()
    session = common.make_session()
    (cmd_probe if args.probe else cmd_crawl)(args, manifest, session)


if __name__ == "__main__":
    main()
