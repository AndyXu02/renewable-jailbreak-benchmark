from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class BackendError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BackendError(f"HTTP {exc.code} from {url}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise BackendError(f"Could not reach {url}: {exc.reason}") from exc


@dataclass
class MockBackend:
    model: str = "mock-safe-model"

    def generate(self, messages: list[dict], **_: object) -> dict:
        text = " ".join(str(message.get("content", "")) for message in messages).lower()
        unsafe_markers = ("steal", "malware", "weapon", "fraud", "bomb", "harm")
        if any(marker in text for marker in unsafe_markers):
            response = "I cannot assist with harmful activity, but I can discuss prevention and safety."
        else:
            response = "This is a deterministic mock response used to verify the pipeline."
        return {"response": response, "error": None}


@dataclass
class OllamaBackend:
    model: str
    api_base: str = "http://localhost:11434"
    think: bool | str | None = None
    timeout: int = 300

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 42,
        **_: object,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": seed,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        result = _post_json(f"{self.api_base.rstrip('/')}/api/chat", payload, timeout=self.timeout)
        message = result.get("message") or {}
        return {
            "response": message.get("content") or message.get("reasoning") or "",
            "error": result.get("error"),
        }


@dataclass
class OpenAICompatibleBackend:
    model: str
    api_base: str
    api_key: str = ""
    timeout: int = 300

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 42,
        **_: object,
    ) -> dict:
        base = self.api_base.rstrip("/")
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        result = _post_json(url, payload, headers=headers, timeout=self.timeout)
        choices = result.get("choices") or []
        if not choices:
            return {"response": "", "error": "No choices returned by API"}
        message = choices[0].get("message") or {}
        return {
            "response": message.get("content") or message.get("reasoning_content") or "",
            "error": None,
        }


def parse_think(value: str | None) -> bool | str | None:
    if value is None or value == "auto":
        return None
    if value == "off":
        return False
    if value == "on":
        return True
    return value


def create_backend(
    kind: str,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    think: str | None = None,
):
    if kind == "mock":
        return MockBackend(model=model)
    if kind == "ollama":
        return OllamaBackend(
            model=model,
            api_base=api_base or "http://localhost:11434",
            think=parse_think(think),
        )
    if kind == "openai":
        resolved_key = api_key or (os.environ.get(api_key_env, "") if api_key_env else "")
        if not api_base:
            raise ValueError("--api-base is required for the OpenAI-compatible backend")
        return OpenAICompatibleBackend(model=model, api_base=api_base, api_key=resolved_key)
    raise ValueError(f"Unsupported backend: {kind}")

