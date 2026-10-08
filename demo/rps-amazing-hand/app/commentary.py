"""Commentator contract + implementations: NullCommentator and a llama.cpp client."""

import json
from pathlib import Path
from typing import AsyncIterator, Protocol

import httpx

import config
from data_model import Match, Outcome, Round


class Commentator(Protocol):
    # False when no model is available. The canned lead-in clips exist only to
    # cover generation latency, so with nothing being generated the game must
    # skip them -- otherwise the hand promises a verdict it never delivers.
    enabled: bool

    async def warm_up(self) -> None: ...

    def opening(self) -> AsyncIterator[str]: ...

    def comment(self, match: Match, rnd: Round) -> AsyncIterator[str]: ...


class NullCommentator:
    enabled = False

    async def warm_up(self) -> None:
        return

    async def opening(self) -> AsyncIterator[str]:
        return
        yield  # pragma: no cover

    async def comment(self, match: Match, rnd: Round) -> AsyncIterator[str]:
        return
        yield  # pragma: no cover


SYSTEM_PROMPT = Path(config.SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
OPENING_PROMPT = Path(config.OPENING_PROMPT_PATH).read_text(encoding="utf-8")


def _describe_round(rnd: Round) -> str:
    machine_outcome = {
        Outcome.MACHINE: "victory",
        Outcome.HUMAN: "defeat",
        Outcome.DRAW: "draw",
    }[rnd.outcome]
    human_move = rnd.human_gesture.value if rnd.human_gesture is not None else "no move"
    return "\n".join(
        [
            "Authoritative completed-round facts:",
            f"Machine move: {rnd.machine_gesture.value}.",
            f"Human opponent move: {human_move}.",
            f"Machine outcome: {machine_outcome}.",
        ]
    )


class LlamaCppCommentator:
    """Streams a one-line remark per round from a llama.cpp-compatible server.

    Any transport, HTTP, or parse failure yields nothing so a dead model never
    interrupts a match.
    """

    enabled = True

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = config.COMMENTARY_TIMEOUT,
        temperature: float = config.COMMENTARY_TEMPERATURE,
        max_tokens: int = config.COMMENTARY_MAX_TOKENS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def warm_up(self) -> None:
        # Prime model initialization and the system-prompt prefix without
        # producing browser-visible commentary or delaying application startup.
        print(f"commentary: using llama.cpp at {self._base_url}")
        print("commentary: warming the LLM ...")
        async for _ in self._stream("Confirm that you are ready.", max_tokens=1):
            pass

    async def opening(self) -> AsyncIterator[str]:
        async for chunk in self._stream(OPENING_PROMPT):
            yield chunk

    async def comment(self, match: Match, rnd: Round) -> AsyncIterator[str]:
        async for chunk in self._stream(_describe_round(rnd)):
            yield chunk

    async def _stream(self, prompt: str, *, max_tokens: int | None = None) -> AsyncIterator[str]:
        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "temperature": self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        url = f"{self._base_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        await response.aread()
                        print(f"commentary: llama-server returned {response.status_code}")
                        return
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"].get("content")
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        if delta:
                            yield delta
        except (httpx.HTTPError, OSError) as exc:
            print(f"commentary: llama-server unreachable ({exc})")
