#!/usr/bin/env python3
"""Compare PDF text extractors on the documents you already have.

For categories that are PDF-only at the source, crawling adds nothing -- the variable
that matters is which extractor you use, because whatever a model ingested was also
PDF-derived by somebody's pipeline. This measures the extractors head to head.

The headline metric is `hard_wrap_pct`: the share of newlines followed by a lowercase
letter, i.e. line breaks that fall mid-sentence. Those are layout artifacts, not
document structure, and they are what corrupts teacher-forced suffix probabilities.
Lower is closer to paragraph-flow text.

  python Memorization/extract/compare_extractors.py --category finra_awc -n 5
"""
from __future__ import annotations
import argparse, importlib, json, os, re, statistics, subprocess, sys, time, warnings, glob, random

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(REPO, "Data Collection and Training Material Generation", "raw_data")
MANIFEST = os.path.join(REPO, "Data Collection and Training Material Generation",
                        "crawl_manifest", "case_ids.json")


def have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def have_bin(name: str) -> bool:
    return subprocess.run(["which", name], capture_output=True).returncode == 0


# ------------------------------------------------------------- extractors
def x_pdfplumber(path):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


def x_pypdf(path):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


def x_pymupdf(path):
    import fitz
    with fitz.open(path) as d:
        return "\n".join(pg.get_text("text") for pg in d)


def x_pdftotext(path):
    r = subprocess.run(["pdftotext", "-q", path, "-"], capture_output=True, timeout=180)
    return r.stdout.decode("utf-8", "replace")


def x_pdftotext_layout(path):
    r = subprocess.run(["pdftotext", "-q", "-layout", path, "-"], capture_output=True, timeout=180)
    return r.stdout.decode("utf-8", "replace")


EXTRACTORS = [
    ("pdfplumber",       x_pdfplumber,       lambda: have("pdfplumber")),
    ("pypdf",            x_pypdf,            lambda: have("pypdf")),
    ("pymupdf",          x_pymupdf,          lambda: have("fitz")),
    ("pdftotext",        x_pdftotext,        lambda: have_bin("pdftotext")),
    ("pdftotext-layout", x_pdftotext_layout, lambda: have_bin("pdftotext")),
]

CAT_DIRS = {
    "finra_awc": ["03_FINRA_AWC/Individual_AWC_Case_Files",
                  "04_RegBI_Compliance_Guidance/FINRA_Reg_BI_AWC_Cases"],
    "arbitration_awards": ["12_FINRA_Arbitration_Awards/Arbitration_Award_Cases"],
    "sec_ia_releases": ["02_SEC_Enforcement_Actions/IA_Act_Enforcement_Releases"],
    "oho_decisions": ["21_FINRA_OHO_Decisions/OHO_Hearing_Decisions_by_Respondent"],
    "court_opinions": ["11_Case_Law/Core_Federal_Court_Opinions"],
}


def stats(t: str) -> dict:
    if not t.strip():
        return {"chars": 0, "hard_wrap_pct": None, "mean_line": None, "hyphen_splits": None}
    nl = re.findall(r"\n(.)", t)
    lines = [l for l in t.split("\n") if l.strip()]
    return {"chars": len(t),
            "hard_wrap_pct": round(100 * sum(1 for c in nl if c.islower()) / max(1, len(nl)), 1),
            "mean_line": round(statistics.mean(len(l) for l in lines), 1),
            "hyphen_splits": len(re.findall(r"[a-z]-\n[a-z]", t))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", action="append", choices=list(CAT_DIRS))
    ap.add_argument("-n", type=int, default=5, help="documents per category")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    avail = [(n, f) for n, f, ok in EXTRACTORS if ok()]
    missing = [n for n, _, ok in EXTRACTORS if not ok()]
    print("available:", ", ".join(n for n, _ in avail) or "none")
    if missing:
        print("missing  :", ", ".join(sorted(set(missing))),
              "\n           (brew install poppler; pip install pymupdf)")

    random.seed(args.seed)
    results = {}
    for cat in (args.category or list(CAT_DIRS)):
        files = []
        for sub in CAT_DIRS[cat]:
            files += glob.glob(os.path.join(RAW, sub, "*.pdf"))
        if not files:
            continue
        sample = random.sample(sorted(files), min(args.n, len(files)))
        print(f"\n=== {cat}  ({len(sample)} docs)")
        print(f"   {'extractor':<18}{'sec/doc':>9}{'chars':>10}{'hard_wrap%':>12}{'mean_line':>11}{'hyph':>7}")
        results[cat] = {}
        for name, fn in avail:
            agg, t0, errs = [], time.time(), 0
            for p in sample:
                try:
                    agg.append(stats(fn(p)))
                except Exception:
                    errs += 1
            if not agg:
                print(f"   {name:<18}{'ERROR':>9}")
                continue
            ok = [a for a in agg if a["hard_wrap_pct"] is not None]
            row = {"sec_per_doc": round((time.time() - t0) / len(sample), 2),
                   "chars": int(statistics.mean(a["chars"] for a in agg)),
                   "hard_wrap_pct": round(statistics.mean(a["hard_wrap_pct"] for a in ok), 1) if ok else None,
                   "mean_line": round(statistics.mean(a["mean_line"] for a in ok), 1) if ok else None,
                   "hyphen_splits": int(statistics.mean(a["hyphen_splits"] for a in ok)) if ok else None,
                   "errors": errs}
            results[cat][name] = row
            print(f"   {name:<18}{row['sec_per_doc']:>9}{row['chars']:>10}"
                  f"{row['hard_wrap_pct']:>12}{row['mean_line']:>11}{row['hyphen_splits']:>7}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extractor_comparison.json")
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    print("\nLower hard_wrap% = closer to paragraph-flow text. Pick by this plus a positive-control\n"
          "check (push known-memorized text through the same pipeline and see which scores highest).")


if __name__ == "__main__":
    main()
