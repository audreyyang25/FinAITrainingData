"""Turn layout-faithful PDF text into paragraph-flow text.

No PDF extractor reflows paragraphs -- they all emit one line per visual line, because
that is what the PDF encodes. So the unwrap has to happen here, after extraction, or
your teacher-forced suffix probabilities get a spurious `\\n` roughly every 11 tokens.

What is kept vs joined:
  joined  - a break inside a sentence (next line starts lowercase, previous line does
            not end in terminal punctuation)
  kept    - paragraph breaks, list markers (A. / 1. / (i) / bullets), headings,
            and any line short enough to be a heading rather than a wrapped line
"""
from __future__ import annotations
import re

TERMINAL = ('.', ':', ';', '?', '!', '"', '"', "'", ')')
LIST_MARK = re.compile(r"^\s*(?:[A-Za-z]\.|\(?[ivxlIVXL]+[\.\)]|\d+[\.\)]|\(\d+\)|[-•*•])\s+")
HEADING = re.compile(r"^\s*(?:[IVXL]+\.|[A-Z][A-Z0-9 ,\.\-'&/\(\)]{2,60})\s*$")
PAGE_NOISE = re.compile(r"^\s*(?:page\s+\d+(\s+of\s+\d+)?|-\s*\d+\s*-|\d{1,4})\s*$", re.I)


def dehyphenate(t: str) -> str:
    return re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", t)


def strip_page_noise(t: str) -> str:
    return "\n".join(l for l in t.split("\n") if not PAGE_NOISE.match(l))


def unwrap(text: str, *, drop_page_noise: bool = True) -> str:
    """Join intra-paragraph hard wraps, preserve structural breaks."""
    t = dehyphenate(text.replace("\r\n", "\n").replace("\r", "\n"))
    if drop_page_noise:
        t = strip_page_noise(t)

    out: list[str] = []
    for raw in t.split("\n"):
        line = raw.rstrip()
        s = line.strip()
        if not s:
            out.append("")
            continue
        if not out or out[-1] == "":
            out.append(s)
            continue
        prev = out[-1]
        starts_structural = bool(LIST_MARK.match(s)) or bool(HEADING.match(s)) or s[0].isupper() and len(prev) < 45
        prev_terminal = prev.endswith(TERMINAL)
        if not starts_structural and not prev_terminal and s[0].islower():
            out[-1] = prev + " " + s
        elif not starts_structural and not prev_terminal and len(prev) > 55:
            out[-1] = prev + " " + s
        else:
            out.append(s)

    res = "\n".join(out)
    res = re.sub(r"[ \t]+", " ", res)
    res = re.sub(r"\n{3,}", "\n\n", res)
    return res.strip()


def hard_wrap_pct(t: str) -> float:
    nl = re.findall(r"\n(.)", t)
    return round(100 * sum(1 for c in nl if c.islower()) / max(1, len(nl)), 1)


if __name__ == "__main__":
    import sys, glob, os, random, warnings
    warnings.filterwarnings("ignore")
    from pypdf import PdfReader
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW = os.path.join(REPO, "Data Collection and Training Material Generation", "raw_data")
    pat = sys.argv[1] if len(sys.argv) > 1 else "03_FINRA_AWC/Individual_AWC_Case_Files"
    random.seed(42)
    for p in random.sample(sorted(glob.glob(os.path.join(RAW, pat, "*.pdf"))), 4):
        raw = "\n".join(pg.extract_text() or "" for pg in PdfReader(p).pages)
        nrm = unwrap(raw)
        print(f"{os.path.basename(p)[:52]:<54} {hard_wrap_pct(raw):>5}% -> {hard_wrap_pct(nrm):>5}%"
              f"   lines {raw.count(chr(10)):>5} -> {nrm.count(chr(10)):>5}")
