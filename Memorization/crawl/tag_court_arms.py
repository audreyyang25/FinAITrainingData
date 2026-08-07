#!/usr/bin/env python3
"""Label court opinions with pre/post-cutoff arms, so all five categories match.

Dates come from CourtListener's `dateFiled` wherever the fetch retrieved it -- that is
the authoritative filing date, not an inference. Filename years are only a fallback for
documents the fetch could not resolve. (Across the fetched set the two agree exactly,
unlike arbitration and AWC where filename years were off by one to three years, so the
fallback is safe here.)

  python Memorization/crawl/tag_court_arms.py
"""
from __future__ import annotations
import collections, datetime, glob, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
MANIFEST = os.path.join(BASE, "crawl_manifest", "case_ids.json")
FETCHED = os.path.join(BASE, "crawl_manifest", "raw", "court_opinions")
CUTOFF = datetime.date(2023, 12, 31)

# The folder is named Core_Federal_Court_Opinions but 21 of 119 are state courts.
# Tag rather than drop: Delaware Chancery alone is 9 of them and is central to
# securities and corporate law, while the rest are single cases from 8 different
# courts -- too thin to support a claim about state courts, fine as labeled rows.
import re
CIRCUIT = re.compile(r"^(ca\d{1,2}|cadc|cafc)$")
DISTRICT = re.compile(r"^[a-z]{2}[ncsewmd]?d$")     # dcd, nysd, cacd, ...


def classify_court(court: str | None):
    c = (court or "").lower()
    if not c:
        return None, None
    if c == "scotus":
        return "federal", "supreme"
    if CIRCUIT.match(c):
        return "federal", "circuit"
    if DISTRICT.match(c) or c in {"dcd", "nysd", "nynd", "nyed", "cacd", "cand", "casd"}:
        return "federal", "district"
    return "state", "state"


def main():
    man = json.load(open(MANIFEST))
    cases = man["court_opinions"]["cases"]

    # case_id -> (date_filed, verdict, match_method) from the fetch
    fetched = {}
    for p in glob.glob(os.path.join(FETCHED, "*.json")):
        try:
            j = json.load(open(p))
        except Exception:
            continue
        cid = (j.get("manifest_record") or {}).get("case_id")
        if cid:
            fetched[cid] = {"date_filed": (j.get("date_filed") or "")[:10] or None,
                            "verdict": j.get("verdict"),
                            "match_method": j.get("match_method"),
                            "match_score": j.get("match_score"),
                            "opinion_id": j.get("opinion_id"),
                            "cluster_id": j.get("cluster_id"),
                            "field": j.get("field")}

    stats = collections.Counter()
    for r in cases:
        juris, level = classify_court(r.get("court"))
        r["jurisdiction"], r["court_level"] = juris, level
        stats[f"juris_{juris or 'unknown'}"] += 1
        f = fetched.get(r["case_id"], {})
        d, src = f.get("date_filed"), "courtlistener"
        if not d:
            y = r.get("year")
            d, src = (f"{y}-07-01" if y and y.isdigit() else None), "filename_year"
        if not d:
            r["arm"], r["date_filed"], r["date_source"] = None, None, "unknown"
            stats["undated"] += 1
            continue
        arm = "pre_cutoff" if datetime.date.fromisoformat(d) <= CUTOFF else "post_cutoff_control"
        r["arm"] = arm
        r["date_filed"] = d
        r["date_source"] = src
        for k in ("verdict", "match_method", "match_score", "opinion_id", "cluster_id", "field"):
            if f.get(k) is not None:
                r[f"cl_{k}" if k in ("verdict", "field") else k] = f[k]
        stats[arm] += 1
        stats[f"date_from_{src}"] += 1

    man["court_opinions"]["cutoff"] = CUTOFF.isoformat()
    man["court_opinions"]["note"] = (
        "full local pool; superset of the original 40 (see in_original_40). "
        "arm from CourtListener dateFiled where available, else filename year.")
    json.dump(man, open(MANIFEST, "w"), indent=1)

    print(f"court_opinions: {len(cases)} cases")
    print(f"   pre_cutoff           {stats['pre_cutoff']}")
    print(f"   post_cutoff_control  {stats['post_cutoff_control']}")
    if stats["undated"]:
        print(f"   undated              {stats['undated']}")
    print(f"   dates from CourtListener: {stats['date_from_courtlistener']}, "
          f"filename fallback: {stats['date_from_filename_year']}")
    print(f"   jurisdiction: federal={stats['juris_federal']}  state={stats['juris_state']}")
    lv = collections.Counter(r.get("court_level") for r in cases)
    print(f"   court_level: {dict(lv)}")
    st = collections.Counter(r["court"] for r in cases if r.get("jurisdiction") == "state")
    print(f"   state courts: {dict(st.most_common())}")
    xt = collections.Counter((r.get("jurisdiction"), r.get("arm")) for r in cases)
    print("   jurisdiction x arm:")
    for k in sorted(xt, key=lambda t: (str(t[0]), str(t[1]))):
        print(f"      {str(k[0]):<8} {str(k[1]):<20} {xt[k]}")

    print("\nall categories:")
    for cat in ("finra_awc", "arbitration_awards", "sec_ia_releases",
                "court_opinions", "oho_decisions"):
        m = man[cat]
        if "cases" in m:
            c = collections.Counter(x.get("arm") or "untagged" for x in m["cases"])
            tot = len(m["cases"])
        else:
            c = {"pre_cutoff": len(m.get("pre_cutoff", [])),
                 "post_cutoff_control": len(m.get("post_cutoff_control", []))}
            tot = sum(c.values())
        print(f"   {cat:<22}{tot:>5}   pre={c.get('pre_cutoff',0):<4} "
              f"post={c.get('post_cutoff_control',0):<4}"
              + (f" untagged={c['untagged']}" if c.get("untagged") else ""))


if __name__ == "__main__":
    main()
