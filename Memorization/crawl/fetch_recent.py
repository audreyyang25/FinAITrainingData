#!/usr/bin/env python3
"""Expand the post-cutoff control arm with recent federal appellate securities opinions.

The control arm is the scarce resource: only 12 cases postdate 2024-12-31, and
2025H1 is empty entirely. Unlike the FINRA categories (robots-disallowed, fixed
pool), CourtListener carries every opinion, so this arm is cheap to grow.

TWO FILTERS DEFINE THE ARM, and both defaults matter:

  * COURT. Federal courts of appeals only -- see the COURTS note below. The
    corpus this feeds is now one court type, so a pre/post comparison is not
    also a circuit/district/state comparison.
  * DATE. --after defaults to 2026-02-28, the latest knowledge cutoff across
    the three newer models, so every case returned is post-cutoff for all of
    them rather than for only some.

Both are overridable; neither should be relaxed without knowing which confound
it reintroduces.

Efficiency matters because of the 250 requests/day quota. Searching by date range
returns many candidates per request, so the cost is ~1 search per page plus 1
fetch per opinion -- roughly 200 new cases per day, versus the ~2 requests/case
the docket-lookup path needed.

New cases are appended to the manifest and written to the CourtListener text
directory, so build_questions.py picks them up with no changes.

  python Memorization/crawl/fetch_recent.py --dry-run
  python Memorization/crawl/fetch_recent.py --max-requests 90
  python Memorization/crawl/fetch_recent.py --after 2024-07-01 --courts ""   # old behaviour
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
# The federal courts of appeals, and nothing else.
#
# WHY THIS IS NOW ENFORCED. This list previously existed but was never passed to
# the search, which ran unfiltered by court -- so the post-cutoff arm drifted to
# 44% state courts against the pre-cutoff arm's 17%. Court level is not a
# nuisance variable here: appellate courts affirm 75-80% of the time while trial
# courts grant or deny, so the disposition base rate differs by arm and part of
# any pre/post drop is case mix rather than lost knowledge. Restricting both arms
# to one court type removes that confound at search time, which is also the
# cheapest place to remove it -- an off-type case costs ~$0.75 to generate,
# judge and label before it gets discarded.
#
# SCOTUS is excluded deliberately. It is federal and appellate but it is not a
# court of appeals, it takes cases on certiorari rather than by right, and its
# disposition mix is not comparable. The corpus holds exactly one SCOTUS case.
# cafc is included for completeness; it hears no securities litigation in
# practice, so it costs one court id in the query and returns nothing.
COURTS = ["ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9",
          "ca10", "ca11", "cadc", "cafc"]

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


# CourtListener's search endpoint enforces a BURST limit well short of the
# hourly quota: a run pacing at one request per 3s was throttled after 6. The
# session's Retry adapter backs off ~1.5s/3s/6s and then raises, which is far too
# short a ceiling for that window -- so a burst limit surfaced as a hard failure
# and, because the whole search loop sat inside one try, took every remaining
# query and date window down with it. The 2026H2 window was never searched at
# all, which is where the newest cases live.
#
# So: treat throttling as a pause, not an error. Sleep past the burst window and
# retry the same request. A genuine quota exhaustion still stops the run, but
# only after COOLDOWN_MAX pauses have failed to clear it.
COOLDOWN_S = 75
COOLDOWN_MAX = 4


class Throttled(RuntimeError):
    """Rate limited. Retryable after a pause, unlike a budget or transport error."""


def get(session, url, budget, cooldowns=None):
    """Fetch with cooldown-and-retry on throttling.

    `cooldowns` is a shared one-element list acting as a run-wide counter, so a
    run cannot spend forever pausing: four pauses that all fail to clear means
    the daily quota is gone, not that the burst window needs another moment.
    """
    while True:
        try:
            return _get_once(session, url, budget)
        except Throttled as e:
            if cooldowns is None or cooldowns[0] >= COOLDOWN_MAX:
                raise
            cooldowns[0] += 1
            print(f"    throttled — pausing {COOLDOWN_S}s "
                  f"(cooldown {cooldowns[0]}/{COOLDOWN_MAX}); {e}")
            time.sleep(COOLDOWN_S)


def _get_once(session, url, budget):
    budget.spend()
    r = common.fetch(session, url, headers=headers())
    if isinstance(r, Exception):
        # The session's retry adapter has 429 in its status_forcelist, so a
        # throttled request is retried three times and surfaces as RetryError
        # rather than a 429 we can read. Its ~10s of backoff is shorter than the
        # burst window, so this is raised as retryable and the caller pauses.
        name = type(r).__name__
        if "Retry" in name or "TooManyRedirects" in name:
            raise Throttled(f"RetryError after {budget.used} requests")
        return None, f"transport: {name}"
    if r.status_code == 429:
        ra = r.headers.get("Retry-After", "?")
        raise Throttled(f"429 after {budget.used} requests; server says resets in {ra}s")
    if r.status_code != 200:
        return None, f"http {r.status_code}"
    try:
        return r.json(), None
    except ValueError:
        return None, "non-json"


def case_key(caption: str, date_filed: str, court: str) -> tuple:
    """Identity of a CASE, as opposed to a cluster or an opinion.

    Caption is folded hard -- lowercased, punctuation stripped, whitespace
    collapsed, common corporate suffixes dropped -- because the same case is
    captioned inconsistently across dockets ("Inc." vs "Inc", "v." vs "v").
    Court and filing date pin it down; two genuinely distinct cases sharing a
    folded caption, a court AND a date is not a case that occurs in practice.
    """
    # Periods are DELETED, not turned into spaces: "B.V." has to fold to "bv" so
    # the suffix list can strip it. Splitting it to "b v" leaves two stray tokens
    # that no suffix rule matches, and the key stops agreeing with the undotted
    # spelling of the same party -- which is the exact variation this is for.
    c = (caption or "").lower().replace(".", "")
    c = re.sub(r"[^a-z0-9 ]+", " ", c)
    c = re.sub(r"\b(inc|llc|ltd|lp|llp|co|corp|corporation|company|na|plc|bv|nv|sa|ag|gmbh)\b",
               " ", c)
    c = re.sub(r"\s+", " ", c).strip()
    return (c, (date_filed or "")[:10], (court or "").lower())


def slugify(caption: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (caption or "").lower()).strip("_")
    return re.sub(r"_+", "_", s)[:46]


def main():
    ap = argparse.ArgumentParser()
    # 2026-02-28 is the LATEST cutoff in the three-model suite (GPT-5.6 Sol);
    # Fable 5 is 2026-01-31 and Gemini 3.1 Pro is 2025-01-31. A case has to
    # postdate all three to be post-cutoff for all three, so the binding date is
    # the maximum, not Fable's. Starting a month earlier would return February
    # cases that are post-cutoff for Fable and pre-cutoff for GPT-5.6 Sol --
    # ambiguous, dropped from the common-set analysis, and paid for anyway.
    ap.add_argument("--after", default="2026-02-28",
                    help="filed_after date; default is the latest cutoff across the "
                         "three newer models, so every hit is post-cutoff for all of them")
    ap.add_argument("--before", default="", help="optional filed_before date")
    ap.add_argument("--courts", default=" ".join(COURTS),
                    help="space-separated CourtListener court ids; defaults to the "
                         "federal courts of appeals. Pass '' to search unfiltered.")
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
    have_cluster = {c.get("cluster_id") for c in cases if c.get("cluster_id")}
    have_key = {case_key(c.get("caption", ""), c.get("date_filed", ""), c.get("court", ""))
                for c in cases}
    scope = args.courts.strip() or "ALL COURTS (unfiltered)"
    print(f"manifest has {len(cases)} court opinions; searching for more filed after {args.after}")
    print(f"  courts: {scope}\n")

    session = common.make_session()
    budget = Budget(args.max_requests)
    # Shared across both phases: the cap is per RUN, not per phase.
    cooldowns = [0]
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
    seen_clusters, seen_keys = set(), set()
    allowed_courts = {c.strip().lower() for c in args.courts.split() if c.strip()}
    off_type, dupes = collections.Counter(), collections.Counter()
    plan = [(q, w) for w in windows(args.after, args.before) for q in QUERIES]
    try:
        for q, (wa, wb) in plan:
            params = {"q": q, "type": "o", "order_by": "dateFiled desc",
                      "filed_after": wa, "filed_before": wb}
            # CourtListener takes a space-separated list of court ids here. This
            # is the filter that was missing: without it the search is scoped by
            # subject matter but not by court, which is how district and state
            # opinions entered a corpus meant to be one court type.
            if args.courts.strip():
                params["court"] = args.courts.strip()
            url = API + "/search/?" + urllib.parse.urlencode(params)
            data, err = get(session, url, budget, cooldowns)
            if err or not data:
                continue
            n = 0
            for hit in data.get("results", []):
                # Belt and braces: the court filter is applied server-side, so a
                # non-appellate hit means the parameter was ignored rather than
                # honoured. Drop it here rather than discover it after paying to
                # judge the case.
                court = (hit.get("court_id") or "").lower()
                if args.courts.strip() and court not in allowed_courts:
                    off_type[court] += 1
                    continue
                cid_cluster = hit.get("cluster_id")
                caption = hit.get("caseName") or ""
                date_filed = (hit.get("dateFiled") or "")[:10]
                key = case_key(caption, date_filed, court)

                # Three dedup layers, because one case can reach us three ways.
                #
                #   cluster  a cluster already ingested, seen again on another query
                #   case     the SAME opinion published under two cluster ids. Real
                #            example: dmarcian v. DMARC Advisor (4th Cir. 2026-07-10)
                #            is one 30-page opinion resolving consolidated appeals
                #            23-1790 and 25-1084, and CourtListener carries it twice,
                #            once per docket. The texts differ only in the docket
                #            stamped on each page header, so they are 99.85% identical
                #            and neither opinion_id nor cluster_id catches it. It
                #            entered the corpus as two cases and sat in a 15-case
                #            post-cutoff arm as 2 of the 15.
                #   opinion  majority, concurrence and dissent are separate opinion
                #            ids on ONE cluster. Taking all of them would file the
                #            same case three times, so take the first and break.
                if cid_cluster and (cid_cluster in seen_clusters or cid_cluster in have_cluster):
                    dupes["cluster"] += 1
                    continue
                if key in seen_keys or key in have_key:
                    dupes["case"] += 1
                    continue

                for o in (hit.get("opinions") or []):
                    oid = o.get("id")
                    if not oid or oid in seen or oid in have_opinion:
                        continue
                    seen.add(oid)
                    if cid_cluster:
                        seen_clusters.add(cid_cluster)
                    seen_keys.add(key)
                    candidates.append({
                        "opinion_id": oid, "cluster_id": cid_cluster,
                        "caption": caption, "court": hit.get("court_id") or "",
                        "date_filed": date_filed,
                        "docket": hit.get("docketNumber") or None,
                    })
                    n += 1
                    break       # one opinion per cluster — see `opinion` above
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
    by_court = collections.Counter(c["court"] for c in candidates)
    print(f"   courts: {dict(sorted(by_court.items(), key=lambda kv: -kv[1]))}")
    if off_type:
        print(f"   ! {sum(off_type.values())} hits dropped as off-type "
              f"(server-side court filter not honoured): {dict(off_type)}")

    if args.dry_run:
        for c in candidates[:12]:
            print(f"   {c['date_filed']}  {c['court']:<8} {c['caption'][:58]}")
        print(f"\n(dry run — would fetch up to {budget.cap - budget.used} of them)")
        return

    # ---- phase 2: fetch text (1 request each) --------------------------------
    added, skipped_short, failed = [], 0, 0
    try:
        for c in candidates:
            op, err = get(session, f"{API}/opinions/{c['opinion_id']}/", budget, cooldowns)
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
