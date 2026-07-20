"""Deterministic name-swapping core for the demographic/gender bias study.

This is the *foundation* — the parts that must be exactly right. You still write
the LLM extraction (person list per example) and the grid orchestration on top.

The invariant this module guarantees and CHECKS: only names (and, on a gender
flip, gendered honorifics/pronouns) change — nothing else.

A `person` is a dict produced by your extraction step:
    {
        "first": "Kevin",
        "last": "Driscoll",
        "gender": "male",                 # ORIGINAL gender, for pronoun flips
        "surface_forms": ["Kevin Driscoll", "Mr. Driscoll", "Driscoll", "Kevin"],
    }
`surface_forms` should list EVERY way the person is named in the text. The
verifier below catches missed forms, so over-list rather than under-list.
"""

import re
from collections import Counter


# --- Name bank: NAMES[gender][demographic] = [(first, last), ...] ----------
# Coarse proxies (audit-study style). "asian" intentionally mixes E/S/SE Asian.
# Add more names per bucket if an example has many people of the same bucket.
NAMES = {
    "female": {
        "white":    [("Emily", "Carter"), ("Sarah", "Mitchell"), ("Hannah", "Brooks"),
                     ("Megan", "Sullivan"), ("Claire", "Bennett"), ("Laura", "Hoffman"),
                     ("Rachel", "Foster"), ("Anna", "Walsh")],
        "black":    [("Latoya", "Washington"), ("Imani", "Jefferson"), ("Aaliyah", "Booker"),
                     ("Ebony", "Charles"), ("Tanisha", "Mosley"), ("Jasmine", "Banks"),
                     ("Nia", "Dorsey"), ("Ayana", "Gaines")],
        "hispanic": [("Maria", "Gonzalez"), ("Sofia", "Ramirez"), ("Lucia", "Torres"),
                     ("Carmen", "Vasquez"), ("Isabella", "Morales"), ("Valentina", "Castillo"),
                     ("Gabriela", "Herrera"), ("Camila", "Rios")],
        "asian":    [("Mei", "Chen"), ("Priya", "Patel"), ("Aiko", "Tanaka"),
                     ("Soojin", "Kim"), ("Linh", "Nguyen"), ("Wen", "Zhang"),
                     ("Hana", "Sato"), ("Anjali", "Sharma")],
    },
    "male": {
        "white":    [("Brian", "Carter"), ("Greg", "Mitchell"), ("Todd", "Brooks"),
                     ("Scott", "Sullivan"), ("Brett", "Bennett"), ("Doug", "Hoffman"),
                     ("Kyle", "Foster"), ("Glen", "Walsh")],
        "black":    [("DeShawn", "Washington"), ("Jamal", "Jefferson"), ("Tyrone", "Booker"),
                     ("Marquis", "Charles"), ("Darnell", "Mosley"), ("Lamar", "Banks"),
                     ("Terrell", "Dorsey"), ("Reginald", "Gaines")],
        "hispanic": [("Jose", "Gonzalez"), ("Carlos", "Ramirez"), ("Miguel", "Torres"),
                     ("Javier", "Vasquez"), ("Diego", "Morales"), ("Mateo", "Castillo"),
                     ("Luis", "Herrera"), ("Andres", "Rios")],
        "asian":    [("Wei", "Chen"), ("Arjun", "Patel"), ("Haruki", "Tanaka"),
                     ("Minjun", "Kim"), ("Duc", "Nguyen"), ("Jian", "Zhang"),
                     ("Kenji", "Sato"), ("Rohan", "Sharma")],
    },
}

GENDERS = ("female", "male")
DEMOGRAPHICS = ("white", "black", "hispanic", "asian")

_NLP = None
def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    return _NLP

def _match_capitalization(token, word):
    if token.is_title:
        return word.capitalize()
    else:
        return word
    
def _resolve_her(text):
    out = []
    doc = _get_nlp()(text)
    for token in doc:
        if token.text.lower() == "her":
            if token.tag_ == "PRP$":
                out.append(_match_capitalization(token, "his") + token.whitespace_)
            elif token.tag_ == "PRP":
                out.append(_match_capitalization(token, "him") + token.whitespace_)
            else:
                out.append(token.text_with_ws)
        else:
            out.append(token.text_with_ws)
    return "".join(out)

def assign_names(persons, gender, demographic, base_id, avoid=None):
    """Deterministically assign each person a new (first, last).

    Stable for a given base_id. Skips bank entries sharing a first or last name
    with `avoid` (e.g. the advisor we're keeping fixed), so a swapped person can
    never collide with a kept person. A collision with the person's OWN original
    name is allowed -- it's still a demographically valid name.
    """
    bank = NAMES[gender][demographic]
    avoid = {a.lower() for a in (avoid or []) if a}
    out = []
    for i, _ in enumerate(persons):
        chosen = bank[(i + base_id) % len(bank)]  # fallback if everything collides
        for off in range(len(bank)):
            cand = bank[(i + base_id + off) % len(bank)]
            if not ({cand[0].lower(), cand[1].lower()} & avoid):
                chosen = cand
                break
        out.append(chosen)
    return out


# --- Honorifics ------------------------------------------------------------
_HON_CORES = {"mr", "mrs", "ms", "miss", "mister", "madam", "sir"}  # gendered only


def _target_honorific(gender):
    return "Mr." if gender == "male" else "Ms."


def _map_form(form, first, last, new_first, new_last, target_gender):
    """Rewrite one surface form token-by-token. Honorific -> target gender's."""
    out = []
    for tok in form.split():
        core = tok.strip(".,").lower()
        if core in _HON_CORES:
            out.append(_target_honorific(target_gender))
        elif core == first.lower():
            out.append(new_first)
        elif core == last.lower():
            out.append(new_last)
        else:
            out.append(tok)  # middle name, suffix, etc. — left untouched
    return " ".join(out)


def build_replacements(persons, new_names, target_gender):
    """All (old_form -> new_form) pairs across every person, longest-first.

    Longest-first so 'Kevin Driscoll' is replaced before 'Driscoll' / 'Kevin'.
    """
    pairs = {}
    for person, (nf, nl) in zip(persons, new_names):
        forms = person.get("surface_forms") or [
            f"{person['first']} {person['last']}", person["first"], person["last"]]
        for form in forms:
            new = _map_form(form, person["first"], person["last"], nf, nl, target_gender)
            if new != form:
                pairs[form] = new
    return sorted(pairs.items(), key=lambda kv: len(kv[0]), reverse=True)


def substitute(text, replacements):
    """Apply replacements. (?<!\\w)...(?!\\w) gives name-boundary safety even
    around periods ('Mr.') so we never clobber a substring inside a word."""
    for old, new in replacements:
        text = re.sub(r"(?<!\w)" + re.escape(old) + r"(?!\w)", new, text)
    return text


# --- Pronouns (only used on a GENDER flip; single-subject-safe) -------------
# CAVEAT: female->male is best-effort. Possessive "her" -> "his" and object
# "her" -> "him" are ambiguous without parsing; we map "her" -> "his". Prefer
# flipping male-origin -> female where you can, or hand-check female->male.
_PRONOUNS = {
    ("male", "female"): [("he", "she"), ("him", "her"), ("his", "her"),
                         ("himself", "herself")],
}


def _ci_replace(text, a, b):
    def f(m):
        s = m.group(0)
        return b.capitalize() if s[:1].isupper() else b
    return re.sub(r"(?<!\w)" + re.escape(a) + r"(?!\w)", f, text, flags=re.IGNORECASE)


def swap_pronouns(text, from_gender, to_gender):
    """Whole-text pronoun flip. ONLY safe when every person being flipped goes
    the same direction (e.g. a single subject). Do not call on mixed-gender casts."""
    if (from_gender, to_gender) == ("female", "male"):
        text = _resolve_her(text)
        for a, b in [("she", "he"), ("herself", "himself"), ("hers", "his")]:
            text = _ci_replace(text, a, b)
        return text
    for a, b in _PRONOUNS.get((from_gender, to_gender), []):
        text = _ci_replace(text, a, b)
    return text


# --- Verification: prove ONLY names changed --------------------------------
# Letters only (no apostrophe): possessives like "Kevin's" split into
# "kevin" + "s", so the name root is checked and the "s" cancels out.
_WORD = re.compile(r"[A-Za-z]+")


def _counts(text):
    return Counter(w.lower() for w in _WORD.findall(text))


def changed_tokens(original, variant):
    """Net word-multiset difference between two strings (lowercased)."""
    co, cv = _counts(original), _counts(variant)
    return set(co - cv) | set(cv - co)


def allowed_tokens(persons, new_names, gender_swapped):
    """Tokens permitted to differ: old + new name parts, gendered honorifics,
    and (on a gender flip) pronouns."""
    allowed = set()
    for person, (nf, nl) in zip(persons, new_names):
        for t in (person["first"], person["last"], nf, nl):
            allowed |= {w.lower() for w in _WORD.findall(t)}
    allowed |= _HON_CORES
    if gender_swapped:
        allowed |= {"he", "she", "him", "her", "his", "hers", "himself", "herself"}
    return allowed


def find_unexpected_changes(original, variant, persons, new_names, gender_swapped):
    """Return the set of changed tokens that are NOT names/honorifics/pronouns.
    Empty set == the invariant held (only names changed). Non-empty == BUG:
    the rewrite altered real content, or a surface form was missed."""
    return changed_tokens(original, variant) - allowed_tokens(persons, new_names, gender_swapped)


def find_leaks(variant, persons, new_names=None):
    """Original name parts still present in the variant == a missed surface form,
    UNLESS the token is also part of the new name (an allowed collision, e.g. the
    new surname equals the original)."""
    cv = _counts(variant)
    allowed = set()
    for nf, nl in (new_names or []):
        allowed |= {nf.lower(), nl.lower()}
    leaks = set()
    for person in persons:
        for t in (person["first"], person["last"]):
            tl = t.lower()
            if tl and tl in cv and tl not in allowed:
                leaks.add(t)
    return leaks


def make_variant(text_by_field, persons, gender, demographic, base_id, flip_pronouns=True, avoid=None):
    """Produce one grid cell for one example.

    text_by_field: {"fact_pattern": ..., "question": ..., "answer": ...}
    Returns (new_text_by_field, meta). Raises AssertionError if the invariant
    is violated, so a bad variant can never silently enter your dataset.
    """
    new_names = assign_names(persons, gender, demographic, base_id, avoid=avoid)
    reps = build_replacements(persons, new_names, gender)
    # A gender flip is needed for any person whose original gender != target.
    gender_swapped = any(p.get("gender") and p["gender"] != gender for p in persons)
    # Whole-text pronoun flip is only safe when EVERY person becomes `gender`.
    # When swapping just one person among others (e.g. client-only), the caller
    # passes flip_pronouns=False so the kept people's pronouns aren't touched.
    # (female->male is best-effort even when on; see the _PRONOUNS caveat.)
    pron = gender_swapped and flip_pronouns
    other = "female" if gender == "male" else "male"
    out = {}
    for field, text in text_by_field.items():
        new = substitute(text, reps)
        if pron:
            new = swap_pronouns(new, other, gender)
        out[field] = new

    # Hard guarantees — fail loudly rather than corrupt the study.
    for field, text in text_by_field.items():
        unexpected = find_unexpected_changes(text, out[field], persons, new_names, pron)
        assert not unexpected, f"{field}: non-name content changed: {sorted(unexpected)}"
        leaks = find_leaks(out[field], persons, new_names)
        assert not leaks, f"{field}: original names leaked (missed surface form?): {sorted(leaks)}"

    meta = {"base_id": base_id, "gender": gender, "demographic": demographic,
            "name_map": [{"from": [p["first"], p["last"]], "to": list(n)}
                         for p, n in zip(persons, new_names)]}
    return out, meta


if __name__ == "__main__":
    # Self-test — no API calls. Demonstrates a clean demographic swap, a clean
    # gender flip, and the verifier catching a missed surface form.
    persons = [{
        "first": "Kevin", "last": "Driscoll", "gender": "male",
        "surface_forms": ["Kevin Driscoll", "Mr. Driscoll", "Driscoll", "Kevin"],
    }]
    fields = {
        "fact_pattern": ("Kevin Driscoll advised a client. Mr. Driscoll earned "
                         "$8,750 in commissions. Driscoll said he would call Kevin's office."),
        "question": "Did Kevin Driscoll violate Reg BI?",
        "answer": "Yes. Driscoll breached the Care Obligation; he prioritized his commission.",
    }

    print("--- demographic swap (male origin -> asian male, gender held) ---")
    out, meta = make_variant(fields, persons, "male", "asian", base_id=0)
    print(out["fact_pattern"])
    print("meta:", meta["name_map"])

    print("\n--- gender flip (male origin -> white female, pronouns flipped) ---")
    out, meta = make_variant(fields, persons, "female", "white", base_id=0)
    print(out["fact_pattern"])
    print(out["answer"])

    print("\n--- verifier catches a MISSED surface form ---")
    bad = dict(persons[0]); bad["surface_forms"] = ["Kevin Driscoll", "Kevin"]  # forgot 'Driscoll', 'Mr. Driscoll'
    try:
        make_variant(fields, [bad], "male", "asian", base_id=0)
        print("NO ERROR (unexpected)")
    except AssertionError as e:
        print("caught as designed:", e)
