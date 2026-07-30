"""rubric.py -- the fixed 20-criterion codebook for binary presence scoring.

Every answer is scored against these SAME 20 criteria by one labeler, so the
axes are model-independent (no invented labels, no canonicalization). "Present"
means the reasoning is substantively ENGAGED, not merely itemized -- this is what
defeats the frame-vs-absent artifact (a model can argue a concept through the
facts without naming it). Criteria and their include/exclude boundaries were
curated to be MECE-ish; boundaries reference neighbors by number.
"""

GLOBAL_INSTRUCTION = (
    "Score PRESENT if the reasoning is substantively engaged -- argued, asserted, "
    "or applied to the facts -- even if not stated as a distinct labeled point. "
    "Score ABSENT only if the reasoning does not appear at all. Judge by "
    "substance, not vocabulary."
)

# id, label, engaged_if, boundary, cues
CRITERIA = [
    (1, "Risk alignment of product with client profile",
     "reasoning evaluates whether the recommended investment is suitable given the client's established objectives, risk tolerance, or financial profile.",
     "evaluates the match between recommendation and client profile -- not establishing or questioning the client's profile itself (#5), not position size or diversification (#2), not time horizon compatibility (#3), and not the product's intrinsic characteristics (#18).",
     "unsuitable for a conservative investor; risk mismatch; recommendation does not fit the client's objectives"),
    (2, "Concentration, position size and diversification",
     "reasoning evaluates position size, single-name or sector concentration, over-weighting, or lack of diversification.",
     "how much of one thing is held -- not whether the product type fits the client (#1).",
     "62% in one sector; over-concentrated; no diversification"),
    (3, "Time horizon, age and investment duration fit",
     "reasoning evaluates whether the investment's expected holding period, maturity, lockup, or investment duration is compatible with the client's intended investment horizon or life stage.",
     "compatibility between investment duration and investment horizon -- not the client's need for readily accessible cash (#14), not general risk suitability (#1), and not age-related vulnerability (#20).",
     "7-year lockup for someone retiring soon; investment horizon is too short; maturity exceeds planned holding period"),
    (4, "Conflicts of interest and conflict disclosure adequacy",
     "reasoning evaluates advisor incentives, compensation, or commissions creating a conflict, or the adequacy of disclosing or mitigating that conflict.",
     "conflicts specifically -- not general adequacy of information conveyed (#7), not fee reasonableness (#9).",
     "commission incentive; self-dealing; undisclosed conflict"),
    (5, "Client objectives, risk tolerance and KYC adequacy",
     "reasoning establishes, infers, questions, or evaluates the adequacy of the client's investment objectives, risk tolerance, financial circumstances, or KYC profile.",
     "establishing or assessing the client profile itself -- not evaluating whether the recommendation matched that profile (#1), not investment experience (#12), and not financial capacity to absorb losses (#6).",
     "client stated capital preservation; inadequate KYC; risk tolerance was never determined"),
    (6, "Affordability / capacity to bear loss",
     "reasoning evaluates whether the client has sufficient financial resources to withstand potential investment losses without jeopardizing their financial well-being.",
     "financial resilience after losses occur -- not stated willingness to take risk (#5), not immediate liquidity needs or cash availability (#14).",
     "cannot afford to lose this money; entire retirement savings at risk; loss would be financially devastating"),
    (7, "Adequacy of information conveyed to client",
     "reasoning evaluates whether the advisor adequately communicated material information necessary for the client to evaluate the recommendation.",
     "the advisor's communication or disclosure behavior (sender side) -- exclude conflict disclosures (#4), fee reasonableness (#9), affirmative misrepresentations (#11), and whether the client actually understood the information (#10).",
     "failed to explain the risks; did not disclose material features; insufficient disclosure"),
    (8, "Applicable standard of care",
     "reasoning evaluates which legal framework governs (FINRA 2111 suitability, Reg BI, or fiduciary), when it attaches, or what it requires.",
     "identifying or attaching the governing standard -- not whether the advisor met it via diligence (#17), not quantitative-suitability mechanics (#16).",
     "Reg BI Care Obligation applies; suitability under 2111; fiduciary duty attaches"),
    (9, "Fee, costs and cost-benefit reasonableness",
     "reasoning evaluates the level or reasonableness of fees or costs, or weighs product cost against benefit.",
     "cost reasonableness -- not trading frequency or turnover (#16), not commission-driven conflict (#4).",
     "high fees erode returns; cost-benefit unfavorable; expensive share class"),
    (10, "Client understanding & informed consent",
     "reasoning evaluates whether the client actually understood the recommendation, appreciated its risks and consequences, or provided informed consent.",
     "the client's comprehension and agreement (receiver side) -- not whether the advisor adequately communicated the information (#7), and not who initiated the transaction (#15).",
     "did not understand the risks; no informed consent; relied on a child to translate"),
    (11, "Affirmative misstatement or misleading conduct",
     "reasoning evaluates the advisor overstating returns, downplaying risk, high-pressure tactics, misleading statements, or falsifying KYC.",
     "affirmative false or misleading acts -- not mere omission or incompleteness of disclosure (#7).",
     "promised to double the fund; downplayed the risk; falsified the risk profile; high-pressure sales"),
    (12, "Investment experience and sophistication",
     "reasoning evaluates the client's investment experience, sophistication, or ability to evaluate complex products.",
     "experience or sophistication specifically -- distinct from general objectives or KYC adequacy (#5).",
     "no investment experience; unsophisticated investor; sophisticated or accredited"),
    (13, "Consideration of reasonable alternatives",
     "reasoning evaluates whether the advisor considered or presented reasonable alternative investments or strategies.",
     "comparison to alternatives -- not the basis for the chosen recommendation (#17).",
     "failed to present conservative alternatives; a cheaper option was available; no alternatives considered"),
    (14, "Liquidity needs and emergency reserves",
     "reasoning evaluates the client's need for accessible cash, emergency reserves, or near-term liquidity.",
     "need for liquid funds -- not the product's lockup duration (#3), not ability to bear loss (#6).",
     "needs the money in 3 years; no emergency reserves; illiquid but needs cash"),
    (15, "Client direction, unsolicited trades and autonomy",
     "reasoning evaluates who initiated the transaction (unsolicited vs recommended), the client's right to direct their own account, or whether a client request affects the obligation.",
     "initiation and autonomy -- not whether the client understood the recommendation (#10).",
     "client insisted; unsolicited trade; client has the right to direct; request does not waive the duty"),
    (16, "Quantitative suitability / excessive trading",
     "reasoning evaluates trading frequency, turnover, churning, or cost-to-equity / quantitative suitability.",
     "trading volume or turnover -- not the level of product fees (#9).",
     "47 trades in nine months; excessive turnover; churning; cost-to-equity ratio"),
    (17, "Reasonable basis, due diligence and documented rationale",
     "reasoning evaluates the advisor's PROCESS of investigating the product or gathering information before recommending -- due-diligence steps, product research, or the presence or absence of a documented rationale.",
     "the investigation process itself -- NOT merely whether the recommendation was ultimately justified or suitable (#1), which every analysis addresses; not identifying the governing legal standard (#8); not considering alternative recommendations (#13); not the product's intrinsic characteristics (#18).",
     "did not research the product; no due diligence on the issuer; no documented rationale for the recommendation"),
    (18, "Product characteristics, mechanics and complexity",
     "reasoning evaluates the investment's intrinsic characteristics, mechanics, structure, complexity, features, or tax treatment.",
     "the properties of the product itself -- not whether those characteristics make it suitable for a particular client (#1), not advisor diligence (#17), and not fees or expenses (#9).",
     "structured-note mechanics; complex derivative; leveraged ETF; tax-deferral provides no benefit inside an IRA"),
    (19, "Firm supervision and recordkeeping integrity",
     "reasoning evaluates firm-level supervision (Rule 3110), approval processes, monitoring, or recordkeeping and documentation integrity.",
     "firm or supervisory-level obligations -- not the individual advisor's rationale documentation (#17).",
     "inadequate supervision; firm approved Level 4 options; recordkeeping failure"),
    (20, "Senior status, vulnerability and diminished capacity",
     "reasoning evaluates whether a client's age, diminished capacity, cognitive impairment, dependency, or other vulnerability requires heightened care or raises concerns about exploitation.",
     "heightened-care obligations arising from vulnerability -- not the relationship between age and investment horizon (#3).",
     "elder exploitation; diminished capacity; vulnerable retiree; client required heightened protections"),
]

IDS = [c[0] for c in CRITERIA]
LABELS = {c[0]: c[1] for c in CRITERIA}


def _block(c):
    cid, label, engaged, boundary, cues = c
    return (f"[{cid}] {label}\n"
            f"    ENGAGED IF: {engaged}\n"
            f"    BOUNDARY: {boundary}\n"
            f"    CUES: {cues}")


RUBRIC_SYSTEM = f"""You score a legal/compliance analysis answer against a fixed \
codebook of 20 reasoning criteria.

{GLOBAL_INSTRUCTION}

For EACH of the 20 criteria, decide whether the answer engages that reasoning.

CRITERIA:
{chr(10).join(_block(c) for c in CRITERIA)}

Return ONLY JSON of this exact shape (all 20 keys "1".."20" required):
{{"scores": {{"1": {{"present": true, "evidence": "<=15-word quote or empty>"}}, \
"2": {{"present": false, "evidence": ""}}, ... , "20": {{"present": false, "evidence": ""}}}}}}
For PRESENT criteria, quote a short phrase from the answer as evidence; for ABSENT, leave evidence empty."""


def user_prompt(answer):
    return f"ANSWER TO SCORE:\n\n{answer}\n\nScore all 20 criteria as JSON."
