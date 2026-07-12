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