"""OpenRouter adapter — one endpoint, any model.

OpenRouter is OpenAI-wire-compatible, so the installed `openai` SDK works against
it with a base_url override; no per-provider SDKs and no per-provider branching.
Model IDs are `vendor/model` (e.g. `anthropic/claude-opus-4.1`, `openai/gpt-4o`,
`google/gemini-2.5-pro`) and are passed straight through — this module has no
model registry, because which models to run is the caller's decision.

Determinism is not uniform across vendors: several reasoning-family models reject
a non-default `temperature` outright. Rather than drop those rows, the call
retries once without it and records `temp_dropped` so the difference is visible
in the output instead of silently changing what was measured.
"""
from __future__ import annotations
import dataclasses, os, time
from pathlib import Path

# Same convention as Data_Analysis/shared/llm.py: the key lives in the repo-root
# .env, not in the environment, so load it rather than requiring an export.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


@dataclasses.dataclass
class Result:
    text: str = ""
    error: str = ""
    finish: str = ""
    in_tokens: int = 0
    out_tokens: int = 0
    latency_s: float = 0.0
    refused: bool = False
    temp_dropped: bool = False
    served_by: str = ""      # OpenRouter reports which upstream provider served it


def _client():
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    # Referer/title are OpenRouter's optional attribution headers.
    return OpenAI(base_url=BASE_URL, api_key=key, default_headers={
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE", "https://localhost/memorization-study"),
        "X-Title": os.environ.get("OPENROUTER_TITLE", "court-opinion-memorization-probe"),
    })


# OpenRouter enables web search three ways: an `:online` model suffix, a
# `plugins: [{"id": "web"}]` body field, and `web_search_options`. None are sent
# here, and the suffix is rejected outright -- a retrieval-enabled run would
# silently turn a memorization probe into a search benchmark, and the results
# would look like strong recall.
def _assert_no_web_search(model: str) -> None:
    if ":online" in model or model.endswith("/online"):
        raise ValueError(
            f"refusing to run {model!r}: the ':online' suffix enables OpenRouter web "
            f"search. Use the bare model slug — this probe must measure recall, not retrieval."
        )


def call(model: str, system: str, user: str, max_tokens: int = 1024,
         temperature: float | None = 0.0, reasoning: dict | None = None) -> Result:
    """`reasoning` maps to OpenRouter's unified reasoning control.

    Minimize it for a recall probe. Test-time deliberation lets a model
    reconstruct a plausible continuation from domain knowledge rather than
    recall the specific opinion -- inference scored as memorization. It also
    makes models incomparable: unequal reasoning budgets across arms is a
    confound in its own right.
    """
    _assert_no_web_search(model)
    t0 = time.time()
    try:
        client = _client()
    except Exception as e:
        return Result(error=f"{type(e).__name__}: {e}"[:400])

    body = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    if temperature is not None:
        body["temperature"] = temperature
    if reasoning is not None:
        body["reasoning"] = reasoning

    dropped = False
    for attempt in (1, 2):
        try:
            resp = client.chat.completions.create(**body)
            break
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            low = msg.lower()
            if attempt == 1 and "reasoning" in low and "reasoning" in body:
                body.pop("reasoning")   # model doesn't accept the control
                continue
            if attempt == 1 and "temperature" in low and "temperature" in body:
                body.pop("temperature")
                dropped = True
                continue
            return Result(error=msg[:400], latency_s=time.time() - t0, temp_dropped=dropped)

    # OpenRouter surfaces upstream failures as a body-level `error` on an HTTP 200.
    err = getattr(resp, "error", None)
    if err:
        return Result(error=f"upstream: {err}"[:400], latency_s=time.time() - t0,
                      temp_dropped=dropped)
    if not getattr(resp, "choices", None):
        return Result(error="no choices returned", latency_s=time.time() - t0,
                      temp_dropped=dropped)

    ch = resp.choices[0]
    text = (ch.message.content or "").strip()
    finish = ch.finish_reason or ""
    u = getattr(resp, "usage", None)
    return Result(
        text=text,
        finish=finish,
        # Vendors signal a policy decline differently; `content_filter` is the
        # wire-level one. Text-level refusals are left to scoring, not guessed at.
        refused=(finish == "content_filter"),
        in_tokens=getattr(u, "prompt_tokens", 0) or 0,
        out_tokens=getattr(u, "completion_tokens", 0) or 0,
        latency_s=time.time() - t0,
        temp_dropped=dropped,
        served_by=getattr(resp, "provider", "") or "",
    )
