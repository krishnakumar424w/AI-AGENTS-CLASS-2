"""Model providers. The agent only knows `generate`. (Given.)"""
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any


class AgentError(Exception):
    """A run could not finish. `retryable` says whether trying again later could work."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class ModelTurn:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    raw: Any = None     # provider-native content, sent back as-is (keeps Gemini's thought signatures)


# `contents` is a plain list the agent builds up:
#   {"role": "user",  "text": str}
#   {"role": "model", "text": str | None, "tool_calls": [{"name", "args"}], "raw": ...}
#   {"role": "tool",  "name": str, "result": dict}


class GroqProvider:
    def __init__(self, model: str):
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=api_key)
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    def _tool_defs(self, tools: list) -> list:
        defs: list = []
        for tool in tools:
            name = getattr(tool, "__name__", None)
            if not name:
                continue
            doc = (tool.__doc__ or "").strip().splitlines()[0] if tool.__doc__ else ""
            defs.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": doc,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            })
        return defs

    def _to_groq_messages(self, system: str, contents: list[dict]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        pending_tool_ids: list[str] = []

        for item in contents:
            role = item["role"]
            if role == "user":
                messages.append({"role": "user", "content": item.get("text", "")})
                pending_tool_ids = []
            elif role == "model":
                assistant_msg: dict = {"role": "assistant"}
                text = item.get("text")
                if text:
                    assistant_msg["content"] = text
                tool_calls = item.get("tool_calls") or []
                if tool_calls:
                    assistant_msg["tool_calls"] = []
                    for call in tool_calls:
                        call_id = f"call_{uuid.uuid4().hex[:12]}"
                        assistant_msg["tool_calls"].append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call.get("args", {}) or {}, separators=(",", ":")),
                            },
                        })
                        pending_tool_ids.append(call_id)
                messages.append(assistant_msg)
            elif role == "tool":
                tool_call_id = pending_tool_ids.pop(0) if pending_tool_ids else f"call_{uuid.uuid4().hex[:12]}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(item.get("result", {}) or {}, separators=(",", ":")),
                })
        return messages

    def generate(self, system: str, contents: list[dict], tools: list) -> ModelTurn:
        from groq import APIStatusError, RateLimitError

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self._to_groq_messages(system, contents),
                tools=self._tool_defs(tools),
                temperature=0,
                tool_choice="auto",
            )
        except RateLimitError as e:
            raise AgentError("provider_rate_limited", "Model quota exhausted. Wait a minute.", True) from e
        except APIStatusError as e:
            if getattr(e, "status_code", None) == 429:
                raise AgentError("provider_rate_limited", "Model quota exhausted. Wait a minute.", True) from e
            if getattr(e, "status_code", None) and getattr(e, "status_code") >= 500:
                raise AgentError("provider_unavailable", "Model provider failed.", True) from e
            raise AgentError("provider_error", str(e), False) from e
        except Exception as e:
            raise AgentError("provider_error", str(e), False) from e

        message = resp.choices[0].message
        text = message.content or None
        calls = []
        for tool_call in (message.tool_calls or []):
            args = tool_call.function.arguments or "{}"
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                parsed = {"value": args}
            calls.append(ToolCall(tool_call.function.name, parsed if isinstance(parsed, dict) else {"value": parsed}))

        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
        return ModelTurn(text=text, tool_calls=calls,
                         tokens_in=tokens_in,
                         tokens_out=tokens_out,
                         raw=message)


class ScriptedProvider:
    """Replays a fixed list of turns. No network, no quota. Used by the tests and --mock."""

    model = "mock"

    def __init__(self, script: list, loop: bool = False):
        self.original, self.script, self.loop = list(script), list(script), loop
        self.calls: list[list[dict]] = []      # what the agent sent on each call

    def generate(self, system: str, contents: list[dict], tools: list) -> ModelTurn:
        self.calls.append([dict(c) for c in contents])
        if not self.script and self.loop:
            self.script = list(self.original)
        if not self.script:
            return ModelTurn(text="(mock) script exhausted")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def default_provider(model: str | None = None):
    if os.environ.get("GROQ_API_KEY"):
        return GroqProvider(model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
    raise AgentError("provider_missing_key", "Set GROQ_API_KEY before running the agent.", False)


def default_mock() -> ScriptedProvider:
    """Calls check_eligibility, which is a given sample, so it works before you write any tools."""
    return ScriptedProvider([
        ModelTurn(text=None, tool_calls=[ToolCall("check_eligibility", {"student_id": "22CS045", "drive_id": 1})],
                  tokens_in=120, tokens_out=12),
        ModelTurn(text="(mock) Yes, you meet all four Zoho rules.", tokens_in=180, tokens_out=14),
    ], loop=True)
