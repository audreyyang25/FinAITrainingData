import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Key lives in the repo-root .env (two levels up), regardless of cwd.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class TruncationError(Exception):
    pass


def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def call_llm(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    retries: int = 5,
) -> str:

    client = get_client()

    last_error = None

    for attempt in range(retries):

        try:

            kwargs = dict(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
                timeout=300,
            )

            if reasoning_effort:
                kwargs["extra_body"] = {
                    "reasoning": {
                        "effort": reasoning_effort
                    }
                }

            response = client.chat.completions.create(
                **kwargs
            )

            choice = response.choices[0]


            if choice.finish_reason != "stop":
                raise TruncationError(
                    f"{model} stopped with "
                    f"{choice.finish_reason}"
                )


            content = choice.message.content

            # Reasoning models sometimes emit whitespace-only visible content
            # with finish_reason == "stop" (the whole budget went to reasoning
            # tokens). This is usually transient, so raise a plain error that
            # the retry loop below will catch, rather than a fatal one.
            if not content or not content.strip():
                raise RuntimeError(
                    f"{model} returned empty/whitespace content"
                )

            return content.strip()


        except TruncationError:
            raise


        except Exception as e:

            last_error = e

            sleep = 2 ** attempt

            print(
                f"{model} failed: {e}. "
                f"Retrying in {sleep}s"
            )

            time.sleep(sleep)


    raise RuntimeError(
        f"{model} failed after retries: {last_error}"
    )