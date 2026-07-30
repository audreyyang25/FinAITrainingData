from pydantic import BaseModel, Field


class Feature(BaseModel):

    feature: str

    importance: int = Field(
        ge=0,
        le=100
    )

    evidence: str


class ExtractionOutput(BaseModel):
    """Third-party feature extraction from an existing answer.

    Answer + a list of Features, with NO sum-to-100 validator. Such a validator
    would encode a self-report *budget* -- it makes sense when a model allocates
    100 points across its own reasoning, but not when a neutral extractor reads
    someone else's answer. Importances are normalized per-model
    downstream, and importance is the low-signal dimension anyway (feature
    SELECTION carries the signal), so an off-100 total is harmless. Enforcing it
    would drop records non-randomly -- e.g. Llama leaves ~7% of answers summing
    to something other than 100 -- biasing the sample toward whichever models
    happen to budget cleanly. Per-feature bounds (0..100) are still enforced by
    Feature; the pipeline clamps to that range before validating.
    """

    answer: str

    features: list[Feature]



class CanonicalFeature(BaseModel):

    feature: str



class CaseCanonicalization(BaseModel):

    superset: list[str]

    # feature id (as string) -> 0-based index into `superset`, or null if the
    # feature was discarded by the relevance filter.
    mapping: dict[str, int | None]



class GlobalCanonicalization(BaseModel):

    vocabulary: list[str]

    mapping: dict[str, str | None]



class GlobalBatchCanonicalization(BaseModel):

    # One incremental step. Each value is either an integer index into the
    # existing vocabulary (reuse), a new global-feature string, or null
    # (discard). Only genuinely new features cost text tokens.
    mapping: dict[str, int | str | None]