from pydantic import BaseModel, Field, field_validator


# LLMs distribute integer weights across features without a running subtotal,
# so landing on exactly 100 is unreliable -- off-by-a-few is normal. Tolerate
# small drift (keep the model's raw numbers); reject only sums far enough off
# to signal a real mistake (wrong scale, miscount).
IMPORTANCE_SUM_TOLERANCE = 20


class Feature(BaseModel):

    feature: str

    importance: int = Field(
        ge=0,
        le=100
    )

    evidence: str


class GenerationOutput(BaseModel):

    answer: str

    features: list[Feature]


    @field_validator("features")
    @classmethod
    def validate_importance_sum(cls, features):

        total = sum(
            f.importance
            for f in features
        )

        if abs(total - 100) > IMPORTANCE_SUM_TOLERANCE:
            raise ValueError(
                f"Importance scores should sum to ~100 "
                f"(within {IMPORTANCE_SUM_TOLERANCE}), got {total}"
            )

        return features



class ExtractionOutput(BaseModel):
    """Third-party feature extraction from an existing answer.

    Same shape as GenerationOutput but WITHOUT the sum-to-100 validator. That
    validator encodes a self-report *budget* -- it makes sense when a model
    allocates 100 points across its own reasoning, but not when a neutral
    extractor reads someone else's answer. Importances are normalized per-model
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