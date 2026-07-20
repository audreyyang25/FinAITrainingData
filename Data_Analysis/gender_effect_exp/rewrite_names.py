"""#2 (LLM version): name-swap variants via a strictly-instructed LLM.

For the first LIMIT records of each dataset type:
  - extract people + ROLES (LLM, cached), and identify the CLIENT by role -- the
    client (retail investor) is NOT always named first, so position is unreliable.
  - SKIP examples with TWO OR MORE named clients (e.g. a jointly-investing couple):
    can't cleanly vary one without the other. A single named client with an unnamed
    or deceased spouse is fine -- on a gender flip the LLM co-flips the spouse's
    gendered relation terms (widower<->widow, his wife<->her husband).
  - for each (gender x demographic) cell, ask the LLM to rename ONLY the client
    to a gender/demographic-appropriate name (avoiding the advisor's name) and
    adjust the client's pronouns/honorifics -- changing nothing else.
  - GATE every cell with the token-diff verifier: reject (retry, then skip) any
    variant that changed a non-name/pronoun WORD, or that leaked the old name.
    (This also catches the LLM renaming the advisor, since the advisor's name is
    not in the allowed set.)

Writes name_variants.jsonl under modified_data/. Resumable on
(type, base_id, gender, demographic).
"""

import os
import sys
import re
import json
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Data_Analysis/ on sys.path
from config import ADAPTERS, DATA_DIR
from name_swap import NAMES, GENDERS, DEMOGRAPHICS, assign_names, changed_tokens, _HON_CORES

MODIFIED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "modified_data")
os.makedirs(MODIFIED_DIR, exist_ok=True)
OUT = os.path.join(MODIFIED_DIR, "name_variants.jsonl")
CACHE = os.path.join(MODIFIED_DIR, "person_roles.jsonl")
LIMIT = 15
# Gender-only pilot: hold demographic constant so male vs female names come from
# the same bank (demographic isn't a variable yet). Set DEMOS = DEMOGRAPHICS later
# to bring in the race dimension.
DEMOS = ("white",)
MODEL = "anthropic/claude-opus-4.8"
MAX_TRIES = 2

oai = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

PRONOUNS = {"he", "she", "him", "her", "his", "hers", "himself", "herself",
            "they", "them", "their", "theirs", "themselves"}

# Gendered spouse/partner terms the LLM MAY co-flip on a gender change (a single
# client's deceased/unnamed spouse), so the relationship stays consistent.
SPOUSE_TERMS = {"husband", "wife", "husbands", "wives", "widow", "widower",
                "widows", "widowers", "spouse", "spouses", "partner", "partners",
                "fiance", "fiancee", "fiances", "fiancees", "bride", "groom",
                "brides", "grooms"}

EXTRACT_SYS = (
    "You analyze a securities-suitability scenario and identify the NAMED people in it. "
    'Return ONLY JSON: {"persons": [{"first": str, "last": str, '
    '"gender": "male"|"female", "role": "client"|"advisor"|"other"}]}. '
    "A CLIENT is a retail customer/investor whose suitability is at issue (NOT the "
    "financial advisor/representative). Label EVERY such investor 'client' -- a "
    "jointly-investing couple has TWO clients. Only list individuals actually NAMED in "
    "the text; a deceased or non-investing spouse mentioned only by relationship (no "
    'name) is not a person to list. Use first/last as written; if only one name part '
    'is given, set the other to "".'
)

SWAP_SYS = (
    'You rename ONE person in a securities scenario. The CLIENT, currently named "{old}", '
    'must be renamed to "{new}", and everything referring to the CLIENT updated to match a '
    "{gender} person: the client's pronouns, honorifics (Mr./Ms./Mrs.), and -- if the client "
    "has a spouse or partner referred to only by relationship, not by their own name -- the "
    "gendered relationship terms, so the relationship stays consistent (e.g. 'his wife' <-> "
    "'her husband', 'widow' <-> 'widower').\n"
    "CRITICAL RULES:\n"
    "- Change ONLY: the client's name; the client's pronouns/honorifics; and gendered "
    "spouse/partner terms tied to the client.\n"
    "- Do NOT change any OTHER NAMED person -- not the financial advisor, not anyone else. "
    "Their names and pronouns stay exactly as written.\n"
    "- Change NOTHING else: no other word, number, fact, punctuation mark, or spacing. "
    "Do not rephrase, reformat, or 'improve' anything.\n"
    "You receive a JSON object mapping ids to text snippets. Return the SAME JSON object "
    "with the same ids, each snippet rewritten per the rules. Output ONLY the JSON object."
)


def collect_text(obj, acc):
    if isinstance(obj, str):
        acc.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            collect_text(x, acc)
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_text(v, acc)
    return acc


def map_strings(obj, fn):
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [map_strings(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: map_strings(v, fn) for k, v in obj.items()}
    return obj


def _json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def extract(rec):
    text = "\n\n".join(collect_text(rec, []))
    resp = oai.chat.completions.create(
        model=MODEL, max_tokens=1500, extra_body={"reasoning": {"effort": "low"}},
        messages=[{"role": "system", "content": EXTRACT_SYS},
                  {"role": "user", "content": text}])
    obj = _json(resp.choices[0].message.content)
    if not obj:
        return None
    persons = []
    for p in obj.get("persons", []):
        first = (p.get("first") or "").strip()
        last = (p.get("last") or "").strip()
        if first or last:
            persons.append({"first": first, "last": last,
                            "gender": (p.get("gender") or "").lower(),
                            "role": (p.get("role") or "").lower()})
    return {"persons": persons}


def pick_client(persons):
    for p in persons:
        if p.get("role") == "client":
            return p
    return None


def mentions(s, person):
    toks = set(re.findall(r"[a-z]+", s.lower()))
    return any(t and t.lower() in toks for t in (person.get("first"), person.get("last")))


def _name_tokens(first, last):
    """Name tokens plus their plural/family form (Chen -> chen, chens)."""
    out = set()
    for t in (first, last):
        for w in re.findall(r"[A-Za-z]+", t or ""):
            out.add(w.lower())
            out.add(w.lower() + "s")
    return out


def cell_problems(old_s, new_s, of, ol, nf, nl, gender_flip):
    """Return (drift, leaked). drift = changed non-name/pronoun words;
    leaked = old client name still present and not equal to the new name."""
    allowed = _name_tokens(of, ol) | _name_tokens(nf, nl) | _HON_CORES | {"s"}
    if gender_flip:
        allowed |= PRONOUNS | SPOUSE_TERMS
    drift = changed_tokens(old_s, new_s) - allowed
    new_toks = set(re.findall(r"[a-z]+", new_s.lower()))
    keep = _name_tokens(nf, nl)
    leaked = {t for t in (of, ol) if t and t.lower() in new_toks and t.lower() not in keep}
    return drift, leaked


def swap_call(payload, old, new, gender):
    resp = oai.chat.completions.create(
        model=MODEL, max_tokens=3000, extra_body={"reasoning": {"effort": "low"}},
        messages=[{"role": "system", "content": SWAP_SYS.format(old=old, new=new, gender=gender)},
                  {"role": "user", "content": json.dumps(payload)}])
    obj = _json(resp.choices[0].message.content)
    return obj if isinstance(obj, dict) else None


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


if __name__ == "__main__":
    done = {(r["type"], r["base_id"], r["gender"], r["demographic"]) for r in load_jsonl(OUT)}
    cache = {(r["type"], r["base_id"]): r for r in load_jsonl(CACHE)}
    cache_f = open(CACHE, "a")
    out_f = open(OUT, "a")

    for name, adapter in ADAPTERS.items():
        records = json.load(open(os.path.join(DATA_DIR, adapter["file"])))[:LIMIT]
        for rec in records:
            base_id = rec["id"]
            bid = base_id if isinstance(base_id, int) else abs(hash(str(base_id))) % 997
            if all((name, base_id, g, d) in done for g in GENDERS for d in DEMOS):
                continue

            info = cache.get((name, base_id))
            if info is None:
                try:
                    got = extract(rec)
                except Exception as e:
                    print(f"EXTRACT FAIL {name}/{base_id}: {e}")
                    continue
                if got is None:
                    print(f"EXTRACT EMPTY {name}/{base_id}")
                    continue
                info = {"type": name, "base_id": base_id, **got}
                cache[(name, base_id)] = info
                cache_f.write(json.dumps(info) + "\n")
                cache_f.flush()

            clients = [p for p in info["persons"] if p.get("role") == "client"]
            if len(clients) >= 2:
                print(f"skip {name}/{base_id}: {len(clients)} named clients (couple confounds signal)")
                continue
            client = clients[0] if clients else None
            if client is None:
                print(f"skip {name}/{base_id}: no client role identified")
                continue

            others = [p for p in info["persons"] if p is not client]
            avoid = [n for p in others for n in (p.get("first"), p.get("last")) if n]
            strings = collect_text(rec, [])
            relevant = [i for i, s in enumerate(strings) if mentions(s, client)]
            if not relevant:
                print(f"skip {name}/{base_id}: client name not found in text")
                continue
            of, ol = client.get("first", ""), client.get("last", "")
            old_name = f"{of} {ol}".strip()

            for g in GENDERS:
                for d in DEMOS:
                    key = (name, base_id, g, d)
                    if key in done:
                        continue
                    nf, nl = assign_names([client], g, d, bid, avoid=avoid)[0]
                    gender_flip = bool(client.get("gender")) and client["gender"] != g

                    newmap, problem = None, None
                    for _ in range(MAX_TRIES):
                        payload = {str(i): strings[i] for i in relevant}
                        got = swap_call(payload, old_name, f"{nf} {nl}", g)
                        if not got or any(str(i) not in got for i in relevant):
                            problem = "bad/incomplete JSON"
                            continue
                        problem = None
                        for i in relevant:
                            drift, leaked = cell_problems(
                                strings[i], got[str(i)], of, ol, nf, nl, gender_flip)
                            if drift or leaked:
                                problem = f"drift={sorted(drift)} leaked={sorted(leaked)}"
                                break
                        if problem is None:
                            newmap = got
                            break
                    if newmap is None:
                        print(f"GATE FAIL {key}: {problem}")
                        continue

                    new_strings = [newmap[str(i)] if i in relevant else s
                                   for i, s in enumerate(strings)]
                    it = iter(new_strings)
                    new_rec = map_strings(rec, lambda _s: next(it))
                    out_f.write(json.dumps({
                        "type": name, "base_id": base_id, "gender": g, "demographic": d,
                        "name_map": [{"from": [of, ol], "to": [nf, nl]}],
                        "gender_flipped": gender_flip,
                        "record": new_rec}) + "\n")
                    out_f.flush()
                    done.add(key)
                    print(f"ok {key} -> {nf} {nl}")
