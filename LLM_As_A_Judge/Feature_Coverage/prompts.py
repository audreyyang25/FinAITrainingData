GENERATION_SYSTEM = """
You are answering a legal/compliance analysis question.

Provide a complete, rigorous analysis of the case.

After your answer, identify the distinct reasoning features that drove your analysis.

A reasoning feature is:
- a factual consideration,
- a legal factor,
- an analytical step,
- or a piece of evidence

that materially contributed to your reasoning.

A reasoning feature is NOT:
- the final conclusion itself,
- a generic statement of the law,
- a restatement of the question,
- a summary of your answer.

The same feature may support opposite conclusions. 
For example, "client age affects suitability analysis" is a valid feature regardless of whether age supports approval or violation.


GRANULARITY RULE

Two features should be separate if they require independent analytical evaluation.

MERGE:

"Client age affects suitability"
"Older clients may have less ability to recover from losses"

because these are part of the same analytical consideration.

SPLIT:

"Client age affects suitability"
"Client wealth affects ability to tolerate losses"

because these require different factual analyses.

Do not split one analytical factor into multiple facts.

Do not merge different analytical factors merely because they relate to the same topic.


IMPORTANCE ALLOCATION

Assign each feature an importance score representing how much that feature contributed to YOUR analysis.

The scores must:

- be integers between 1 and 100
- sum exactly to 100
- reflect importance relative to other features in THIS answer

Calibration:

90-100:
A dominant driver of the analysis

60-89:
A major reasoning factor

30-59:
A meaningful supporting factor

10-29:
A minor consideration

1-9:
Mentioned but minimally influential


IMPORTANT:

Importance measures contribution to reasoning, not whether the feature supports your conclusion.

A feature supporting a conclusion should not automatically receive higher importance.


OUTPUT FORMAT

Return ONLY valid JSON:

{
  "answer": "...",
  "features": [
    {
      "feature": "...",
      "importance": 40,
      "evidence": "..."
    }
  ]
}

No markdown.
No explanation outside JSON.
"""

CASE_CANONICALIZATION_SYSTEM = """
You are constructing a case-level reasoning feature space.

You will receive a NUMBERED list of reasoning features, extracted from multiple answers to the same legal/compliance question. Different numbers may express the same underlying reasoning.

Your task:

1. Merge features that represent the same analytical step into one canonical feature.
2. Preserve genuinely distinct analytical factors as separate canonical features.
3. Discard features that are not real reasoning steps.

The canonical features (the "superset") become the denominator for a coverage metric, so judge similarity by ANALYTICAL meaning, not wording.


MERGE RULE

Merge two features ONLY if they require the same analytical evaluation.

Examples:

MERGE:

"Client age affects suitability"
"Older clients may have less ability to recover losses"

because both represent the same analytical factor.


DO NOT MERGE:

"Client age affects suitability"

with:

"Client wealth affects ability to tolerate losses"

because they require different analysis.


IMPORTANT:

Semantic similarity is not enough.

Two features can use similar words while representing different reasoning steps.


RELEVANCE FILTER

Discard (map to null):
- conclusions
- generic legal statements
- duplicated restatements
- unsupported tangents


CANONICAL FEATURE FORMAT

Each canonical feature should be:

- stance neutral
- self-contained
- understandable without context
- specific enough to represent a distinct reasoning step


OUTPUT ONLY JSON, no markdown:

{
 "superset":[
   "canonical feature at index 0",
   "canonical feature at index 1"
 ],

 "mapping":{
   "<feature number>": <0-based index of its canonical feature in "superset">
 }
}

MAPPING RULES:
- Include EVERY input feature number as a key (as a string).
- Each value is the 0-based index of that feature's canonical feature in your "superset" array.
- Feature numbers that merge together MUST map to the SAME index.
- If a feature is discarded by the relevance filter, map it to null.
"""