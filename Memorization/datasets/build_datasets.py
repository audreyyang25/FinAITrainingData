#!/usr/bin/env python3
"""Build the two dataset shapes from the local PDFs.

  CSV  (section-per-column)  finra_awc, arbitration_awards, sec_ia_releases
  TXT  (one file per doc)    court_opinions, oho_decisions

Document set comes from crawl_manifest/case_ids.json, so the pre/post-cutoff arms are
carried through as an `arm` column. Output is *raw* extraction -- run clean_newlines.py
afterwards to collapse layout line-wrapping.

  python Memorization/datasets/build_datasets.py
  python Memorization/datasets/build_datasets.py --category finra_awc
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
from pypdf import PdfReader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO, "Data Collection and Training Material Generation")
RAW = os.path.join(BASE, "raw_data")
MANIFEST = os.path.join(BASE, "crawl_manifest", "case_ids.json")
OUT = os.path.join(BASE, "datasets")

CSV_CATS = ["finra_awc", "arbitration_awards", "sec_ia_releases"]
TXT_CATS = ["court_opinions", "oho_decisions"]

SEARCH_DIRS = [
    "03_FINRA_AWC/Individual_AWC_Case_Files",
    "03_FINRA_AWC/Gap_Filled_AWC_Sources",
    "04_RegBI_Compliance_Guidance/FINRA_Reg_BI_AWC_Cases",
    "12_FINRA_Arbitration_Awards/Arbitration_Award_Cases",
    "02_SEC_Enforcement_Actions/IA_Act_Enforcement_Releases",
    "11_Case_Law/Core_Federal_Court_Opinions",
    "21_FINRA_OHO_Decisions/OHO_Hearing_Decisions_by_Respondent",
]


def build_index() -> dict:
    idx = {}
    for sub in SEARCH_DIRS:
        for p in glob.glob(os.path.join(RAW, sub, "*.pdf")) + glob.glob(os.path.join(RAW, sub, "*.PDF")):
            idx.setdefault(os.path.basename(p), p)
    return idx


# ------------------------------------------------------------------ extraction
def pages_of(path):
    return [(p.extract_text() or "") for p in PdfReader(path).pages]


def strip_running(pages):
    """Remove page furniture: lines repeating on >=40% of pages, plus known patterns."""
    if len(pages) >= 3:
        cnt = collections.Counter()
        for pg in pages:
            for ln in {l.strip() for l in pg.splitlines() if l.strip()}:
                cnt[ln] += 1
        thresh = max(2, int(0.4 * len(pages)))
        repeated = {l for l, c in cnt.items() if c >= thresh and len(l) < 90}
    else:
        repeated = set()
    pats = [re.compile(p, re.I) for p in (
        r"^\s*(page\s+)?\d+\s*$", r"^\s*-\s*\d+\s*-\s*$", r"^\s*page\s+\d+\s+of\s+\d+\s*$",
        r"^\s*award\s+page\s+\d+\s+of\s+\d+\s*$", r"^\s*arbitration\s+no\.?\s*[\d\-]+\s*$",
        r"^\s*finra\s+(office\s+of\s+dispute\s+resolution|dispute\s+resolution\s+services)\s*$",
    )]
    out = []
    for pg in pages:
        keep = []
        for ln in pg.splitlines():
            s = ln.strip()
            if not s:
                keep.append("")
                continue
            if s in repeated or any(p.match(s) for p in pats):
                continue
            keep.append(ln)
        out.append("\n".join(keep))
    return "\n".join(out)


def norm(t):
    t = t.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


# ------------------------------------------------------------------- splitters
ARB_HEADS = {"representation of parties", "case information", "case summary", "relief requested",
             "other issues considered and decided", "award", "fees", "arbitration panel",
             "arbitration panels", "concurring opinion", "dissenting opinion",
             "concurring and dissenting opinion"}
AWC_HEADS = {"acceptance and consent", "waiver of procedural rights", "other matters", "background",
             "overview", "facts and violative conduct", "sanctions", "relevant disciplinary history",
             "other factors considered"}
SEC_SUB = {"summary", "respondent", "respondents", "facts", "background", "violations", "findings",
           "undertakings", "disgorgement", "remedial efforts", "other relevant conduct",
           "legal discussion"}
ROMAN = re.compile(r"^\s*([IVXL]+)\.\s*(.{0,70})$")


def slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_"))[:60]


def arb_head(s):
    k = re.sub(r"\s+", " ", s).strip().rstrip(":").lower()
    return k if k in ARB_HEADS and s.isupper() else None


def awc_head(s):
    s2 = re.sub(r"^[IVX]+\.\s*", "", re.sub(r"\s+", " ", s).strip()).rstrip(":.")
    k = s2.lower()
    return k if k in AWC_HEADS and (s2.isupper() or s2.istitle()) else None


def roman_head(s):
    m = ROMAN.match(s)
    if not m:
        return None
    t = m.group(2).strip().rstrip(".")
    return f"{m.group(1).lower()}_{slug(t)}" if t else m.group(1).lower()


def sec_head(s):
    r = roman_head(s)
    if r:
        return r
    k = re.sub(r"\s+", " ", s).strip().rstrip(":").lower()
    return k if k in SEC_SUB and len(s) < 40 else None


SPLIT = {"finra_awc": awc_head, "arbitration_awards": arb_head, "sec_ia_releases": sec_head}


def split_sections(text, is_head):
    secs, cur, buf = [], "_preamble", []
    for ln in text.splitlines():
        h = is_head(ln.strip())
        if h is not None:
            secs.append((cur, "\n".join(buf).strip()))
            cur, buf = h, []
        else:
            buf.append(ln)
    secs.append((cur, "\n".join(buf).strip()))
    return [(h, b) for h, b in secs if b or h != "_preamble"]


# ------------------------------------------------------------------------ main
def iter_cases(man, cat):
    m = man[cat]
    if "cases" in m:
        for r in m["cases"]:
            yield r, "carried_over"
    else:
        for arm in ("pre_cutoff", "post_cutoff_control"):
            for r in m.get(arm, []):
                yield r, arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", action="append", choices=CSV_CATS + TXT_CATS)
    args = ap.parse_args()
    cats = args.category or (CSV_CATS + TXT_CATS)

    man = json.load(open(MANIFEST))
    idx = build_index()
    os.makedirs(OUT, exist_ok=True)

    for cat in cats:
        cases = list(iter_cases(man, cat))
        print(f"\n=== {cat}: {len(cases)} docs", flush=True)
        rows, cols, missing, empty = [], collections.Counter(), 0, 0

        # A case_id is not always unique per document: a FINRA disciplinary proceeding
        # can produce several decisions and orders under one proceeding number. Suffix
        # collisions so nothing gets silently overwritten.
        id_counts = collections.Counter(r["case_id"] for r, _ in cases)
        seen_ids: collections.Counter = collections.Counter()

        if cat in TXT_CATS:
            d = os.path.join(OUT, cat)
            os.makedirs(d, exist_ok=True)
            index = []

        for i, (rec, arm) in enumerate(cases, 1):
            path = idx.get(rec.get("source_file", ""))
            if not path:
                missing += 1
                continue
            try:
                pgs = pages_of(path)
                txt = norm(strip_running(pgs))
            except Exception as e:
                print(f"   ERR {rec['case_id']}: {e}", flush=True)
                continue
            if len(txt) < 200:
                empty += 1

            cid = rec["case_id"]
            seen_ids[cid] += 1
            uid = cid if id_counts[cid] == 1 else f"{cid}__{seen_ids[cid]}"

            if cat in TXT_CATS:
                safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in uid)
                fname = safe + ".txt"
                with open(os.path.join(OUT, cat, fname), "w") as fh:
                    fh.write(txt)
                index.append({"file": fname, "case_id": cid, "doc_uid": uid,
                              "source_file": rec["source_file"], "n_pages": len(pgs),
                              "n_chars": len(txt)})
                rows.append({"case_id": cid, "n_chars": len(txt)})
            else:
                secs = split_sections(txt, SPLIT[cat])
                row = {"case_id": cid, "doc_uid": uid, "arm": arm, "date": rec.get("date"),
                       "source_file": rec["source_file"], "n_pages": len(pgs),
                       "n_chars": len(txt), "n_sections": len(secs)}
                seen = collections.Counter()
                for h, b in secs:
                    seen[h] += 1
                    key = h if seen[h] == 1 else f"{h}__{seen[h]}"
                    row[key] = b
                    cols[key] += 1
                rows.append(row)
            if i % 25 == 0:
                print(f"   {i}/{len(cases)}", flush=True)

        if cat in TXT_CATS:
            json.dump(index, open(os.path.join(OUT, cat, "_index.json"), "w"), indent=1)
            dups = sum(1 for r in index if "__" in r["doc_uid"])
            print(f"   -> {OUT}/{cat}/  {len(rows)} .txt files"
                  + (f" ({dups} disambiguated by suffix)" if dups else "")
                  + (f", {missing} missing" if missing else "")
                  + (f", {empty} near-empty" if empty else ""))
        else:
            meta = ["case_id", "doc_uid", "arm", "date", "source_file", "n_pages", "n_chars", "n_sections"]
            ordered = [c for c, _ in cols.most_common() if c not in meta]
            fp = os.path.join(OUT, f"{cat}.csv")
            with open(fp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=meta + ordered, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f"   -> {fp}  {len(rows)} rows x {len(meta)+len(ordered)} cols"
                  f"{f', {missing} missing' if missing else ''}")
    print(f"\nOutput: {OUT}\nNext: python Memorization/datasets/clean_newlines.py")


if __name__ == "__main__":
    main()
