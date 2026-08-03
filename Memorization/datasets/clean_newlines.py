#!/usr/bin/env python3
"""Collapse layout line-wrapping so each paragraph is one continuous line.

A PDF stores glyph positions, not line breaks. Every newline you get from extraction is
a *visual* line ending -- where the text happened to hit the right margin. Those are not
in the document, and at ~one every 11 tokens they wreck teacher-forced suffix
probabilities.

Intentional breaks are kept:
  - blank-line paragraph separations
  - numbered / lettered / bulleted items:  A.  1.  (i)  (3)  •  -
  - headings (ALL CAPS lines, roman numerals)
  - a break after a line that is clearly short of the margin (end of paragraph)

Everything else is joined. The signal for "this was a wrap, not a break" is that the
previous line runs close to full width -- a genuine paragraph end usually leaves the
last line short. That test is more reliable than punctuation, because a wrap can fall
right after a sentence-ending period and look like a paragraph break when it isn't.

  python Memorization/datasets/clean_newlines.py                 # CSVs -> *_clean.csv
  python Memorization/datasets/clean_newlines.py --txt           # also the .txt dirs
  python Memorization/datasets/clean_newlines.py --inspect finra_awc
"""
from __future__ import annotations
import argparse, csv, glob, os, re, statistics, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "Data Collection and Training Material Generation", "datasets")

CSV_CATS = ["finra_awc", "arbitration_awards", "sec_ia_releases"]
TXT_CATS = ["court_opinions", "oho_decisions"]
META = {"case_id", "arm", "date", "source_file", "n_pages", "n_chars", "n_sections"}

# A line at least this long is assumed to have been broken by the margin, not by intent.
WRAP_MIN = 55

LIST_MARK = re.compile(r"""^\s*(
      [A-Za-z][\.\)]\s          |   # A.  b)
      \(?[ivxlcIVXLC]+[\.\)]\s  |   # (i)  IV.
      \d{1,3}[\.\)]\s           |   # 1.  12)
      \(\d{1,3}\)\s             |   # (3)
      [-•‣●\*]\s     # bullets
    )""", re.X)
HEADING = re.compile(r"^\s*(?:[IVXL]+\.\s*)?[A-Z][A-Z0-9 ,\.\-'&/\(\)]{2,60}\s*$")
PAGE_NOISE = re.compile(r"^\s*(?:page\s+\d+(?:\s+of\s+\d+)?|-\s*\d+\s*-|\d{1,4})\s*$", re.I)
FOOTNOTE = re.compile(r"^\s*\d{1,3}\s+[A-Z]")   # "12 See In re ..." footnote body


def dehyphenate(t: str) -> str:
    return re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", t)


def clean(text: str, *, wrap_min: int = WRAP_MIN, drop_noise: bool = True) -> str:
    if not text or not text.strip():
        return text or ""
    t = dehyphenate(text.replace("\r\n", "\n").replace("\r", "\n"))
    out: list[str] = []
    for raw in t.split("\n"):
        s = raw.strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue
        if drop_noise and PAGE_NOISE.match(s):
            continue
        if not out or out[-1] == "":
            out.append(s)
            continue
        prev = out[-1]
        structural = bool(LIST_MARK.match(s)) or bool(HEADING.match(s)) or bool(FOOTNOTE.match(s))
        prev_is_heading = bool(HEADING.match(prev))
        # Join when the previous line ran to the margin and this line is not a new
        # structural element. Punctuation is deliberately not consulted.
        if not structural and not prev_is_heading and len(prev) >= wrap_min:
            out[-1] = prev + " " + s
        else:
            out.append(s)
    res = "\n".join(out)
    res = re.sub(r"[ \t]+", " ", res)
    return re.sub(r"\n{3,}", "\n\n", res).strip()


def wrap_pct(t: str) -> float:
    nl = re.findall(r"\n(.)", t or "")
    return round(100 * sum(1 for c in nl if c.islower()) / max(1, len(nl)), 1)


def clean_csv(cat: str) -> dict:
    src = os.path.join(DATA, f"{cat}.csv")
    dst = os.path.join(DATA, f"{cat}_clean.csv")
    if not os.path.exists(src):
        return {"category": cat, "error": "missing"}
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
        cols = rows[0].keys() if rows else []
    before = after = 0
    b_lines = a_lines = 0
    for r in rows:
        for c in cols:
            if c in META or not r.get(c):
                continue
            v = r[c]
            b_lines += v.count("\n")
            before += len(re.findall(r"\n(.)", v))
            r[c] = clean(v)
            a_lines += r[c].count("\n")
            after += len(re.findall(r"\n(.)", r[c]))
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols))
        w.writeheader()
        w.writerows(rows)
    return {"category": cat, "rows": len(rows), "newlines_before": b_lines,
            "newlines_after": a_lines,
            "reduction_pct": round(100 * (1 - a_lines / max(1, b_lines)), 1), "out": dst}


def clean_txt(cat: str) -> dict:
    d = os.path.join(DATA, cat)
    dd = os.path.join(DATA, f"{cat}_clean")
    if not os.path.isdir(d):
        return {"category": cat, "error": "missing"}
    os.makedirs(dd, exist_ok=True)
    b = a = n = 0
    for p in sorted(glob.glob(os.path.join(d, "*.txt"))):
        t = open(p).read()
        c = clean(t)
        open(os.path.join(dd, os.path.basename(p)), "w").write(c)
        b += t.count("\n"); a += c.count("\n"); n += 1
    return {"category": cat, "files": n, "newlines_before": b, "newlines_after": a,
            "reduction_pct": round(100 * (1 - a / max(1, b)), 1), "out": dd}


def inspect(cat: str, n: int = 2):
    src = os.path.join(DATA, f"{cat}.csv")
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    shown = 0
    for r in rows:
        for c, v in r.items():
            if c in META or not v or len(v) < 700:
                continue
            print(f"\n=== {r['case_id']}  [{c}]")
            print("--- BEFORE ---"); print(v[:600])
            print("--- AFTER ----"); print(clean(v)[:600])
            shown += 1
            break
        if shown >= n:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", action="store_true", help="also clean the .txt directories")
    ap.add_argument("--category", action="append", choices=CSV_CATS + TXT_CATS)
    ap.add_argument("--inspect", metavar="CATEGORY", help="print before/after samples, write nothing")
    ap.add_argument("--wrap-min", type=int, default=WRAP_MIN)
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    print(f"{'category':<22}{'units':>7}{'newlines':>12}{'->':>8}{'reduction':>12}")
    for cat in (args.category or CSV_CATS):
        if cat in CSV_CATS:
            r = clean_csv(cat)
            if r.get("error"):
                print(f"{cat:<22}  {r['error']}")
                continue
            print(f"{cat:<22}{r['rows']:>7}{r['newlines_before']:>12}{r['newlines_after']:>8}"
                  f"{str(r['reduction_pct'])+'%':>12}")
    if args.txt or (args.category and any(c in TXT_CATS for c in args.category)):
        for cat in (args.category or TXT_CATS):
            if cat in TXT_CATS:
                r = clean_txt(cat)
                if r.get("error"):
                    print(f"{cat:<22}  {r['error']}")
                    continue
                print(f"{cat:<22}{r['files']:>7}{r['newlines_before']:>12}{r['newlines_after']:>8}"
                      f"{str(r['reduction_pct'])+'%':>12}")
    print(f"\nOutput alongside originals in {DATA} (suffix _clean).")


if __name__ == "__main__":
    main()
