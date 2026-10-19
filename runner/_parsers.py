"""Streaming-friendly parser that pulls structured fields out of raw model output.

Recognised templates:

  ``<think>...</think>``           -> ``reasoning_content`` (DeepSeek-R1, Qwen3
                                       with ``enable_thinking=True``, GPT-OSS)

  ``<tool_call>{...}</tool_call>`` -> ``tool_calls`` (Hermes 3 format used by
                                       Qwen 2.5 / Qwen 3 / Hermes-2-Pro and
                                       most Hermes-trained checkpoints)

This is the same job OpenVINO Model Server does with
``--tool_parser hermes3 --reasoning_parser qwen3``, but kept as a small
self-contained Python module so we don't take OVMS as a dependency or change
illama's "FastAPI + OpenVINO GenAI" backbone.

Why streaming-friendly?
~~~~~~~~~~~~~~~~~~~~~~~

The runner emits SSE / NDJSON deltas as tokens arrive. A naive split on
``</think>`` can't run on each delta because the closing tag may straddle two
deltas (one delta ends with ``</thi``, the next starts with ``nk>``). The
:class:`StreamingParser` keeps a small hold-back buffer (long enough to recognise
any of the supported tags) and only emits text once we know it isn't part of a
boundary.

API summary
~~~~~~~~~~~

* :func:`parse_full_output` - one-shot helper for non-streaming endpoints.
  Returns a dict with ``content``, optional ``reasoning_content``, and optional
  ``tool_calls`` (OpenAI-shaped).

* :class:`StreamingParser` - feed text chunks via :meth:`feed`, drain at end via
  :meth:`finalize`. Both return a list of ``(kind, payload)`` tuples that the
  caller turns into deltas. ``kind`` is one of:

    - ``"content"`` -> ``str`` (assistant text delta)
    - ``"reasoning"`` -> ``str`` (reasoning delta, accumulate into
      ``reasoning_content``)
    - ``"tool"`` -> ``dict`` (a complete OpenAI-shaped ``tool_calls[i]`` entry,
      emitted once the ``</tool_call>`` tag closes; clients buffer these
      regardless so atomic emission keeps the parser simple)

Fail-soft policy: malformed JSON inside ``<tool_call>`` is dropped silently
(the raw text was already withheld from ``content``). A surfaced parse error
would force every model that emits Hermes-style JSON to be perfectly
syntactically valid, which they aren't.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Tag table. Order matters only to keep the docstring readable; ``feed()``
# picks the earliest match in the buffer regardless of declaration order.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

_ALL_TAGS = (THINK_OPEN, THINK_CLOSE, TOOL_OPEN, TOOL_CLOSE)
# How many trailing characters of the buffer we keep unflushed in case they
# turn out to be the prefix of a tag. ``len(longest_tag) - 1`` is the tightest
# safe bound: any partial tag short of full match is at most this long.
_HOLDBACK = max(len(t) for t in _ALL_TAGS) - 1


ParserEvent = Tuple[str, Any]


def parse_full_output(text: str) -> Dict[str, Any]:
    """Run the parser end-to-end on a complete generation.

    Returns a dict with at least ``content``; ``reasoning_content`` and
    ``tool_calls`` are present only when the model emitted them.
    """
    parser = StreamingParser()
    events = parser.feed(text) + parser.finalize()

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    for kind, payload in events:
        if kind == "content":
            content_parts.append(payload)
        elif kind == "reasoning":
            reasoning_parts.append(payload)

    out: Dict[str, Any] = {"content": "".join(content_parts)}
    reasoning = "".join(reasoning_parts).strip()
    if reasoning:
        out["reasoning_content"] = reasoning
    if parser.tool_calls:
        out["tool_calls"] = list(parser.tool_calls)
    return out


class StreamingParser:
    """Stateful parser that splits raw model output into deltas.

    Construct one parser per generation. Call :meth:`feed` for each text chunk
    the streamer hands you, then :meth:`finalize` once when generation ends.
    Both return event lists ready to be turned into wire deltas.

    The parser is *not* threadsafe; each generation runs in a single worker
    thread, so we don't need locking.
    """

    def __init__(self) -> None:
        self._buf: str = ""
        # Possible values: "content", "reasoning", "tool".
        self._mode: str = "content"
        self._tool_acc: str = ""
        self._tool_index: int = 0
        # Mirrors any tool_call we emitted, so callers can build the final
        # non-streaming response without re-parsing.
        self.tool_calls: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, text: str) -> List[ParserEvent]:
        """Append ``text`` to the buffer and emit any deltas it unblocks."""
        if not text:
            return []
        self._buf += text
        return self._drain(final=False)

    def finalize(self) -> List[ParserEvent]:
        """Flush whatever's left when the model is done generating.

        After this returns, no more events will be emitted.
        """
        return self._drain(final=True)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _drain(self, final: bool) -> List[ParserEvent]:
        events: List[ParserEvent] = []
        while self._buf:
            if self._mode == "content":
                if not self._drain_content(events, final):
                    break
            elif self._mode == "reasoning":
                if not self._drain_reasoning(events, final):
                    break
            elif self._mode == "tool":
                if not self._drain_tool(events, final):
                    break
            else:  # pragma: no cover - defensive, _mode is set internally
                # Unknown state: fall back to dumping buffer as content so we
                # never silently swallow tokens.
                events.append(("content", self._buf))
                self._buf = ""
                self._mode = "content"
        return events

    def _drain_content(self, events: List[ParserEvent], final: bool) -> bool:
        """Returns True if the loop should keep going, False to break."""
        idx_think = self._buf.find(THINK_OPEN)
        idx_tool = self._buf.find(TOOL_OPEN)
        candidates = [i for i in (idx_think, idx_tool) if i >= 0]

        if candidates:
            tag_at = min(candidates)
            if tag_at > 0:
                events.append(("content", self._buf[:tag_at]))
                self._buf = self._buf[tag_at:]
            if self._buf.startswith(THINK_OPEN):
                self._buf = self._buf[len(THINK_OPEN):]
                self._mode = "reasoning"
            else:
                # Must be TOOL_OPEN at this point.
                self._buf = self._buf[len(TOOL_OPEN):]
                self._mode = "tool"
                self._tool_acc = ""
            return True

        # No tag visible. Flush everything we can without breaking a tag boundary.
        return self._flush_safe(events, final, kind="content")

    def _drain_reasoning(self, events: List[ParserEvent], final: bool) -> bool:
        idx = self._buf.find(THINK_CLOSE)
        if idx >= 0:
            if idx > 0:
                events.append(("reasoning", self._buf[:idx]))
            self._buf = self._buf[idx + len(THINK_CLOSE):]
            self._mode = "content"
            return True
        return self._flush_safe(events, final, kind="reasoning")

    def _drain_tool(self, events: List[ParserEvent], final: bool) -> bool:
        idx = self._buf.find(TOOL_CLOSE)
        if idx >= 0:
            self._tool_acc += self._buf[:idx]
            self._buf = self._buf[idx + len(TOOL_CLOSE):]
            self._mode = "content"
            tool_call = self._build_tool_call(self._tool_acc)
            self._tool_acc = ""
            if tool_call is not None:
                events.append(("tool", tool_call))
            return True

        if final:
            # Stream ended mid-tool-call. Try to parse what we have; if it
            # doesn't decode, drop it on the floor (no content emit either,
            # since the user asked for a tool call and we'd be lying to
            # surface garbage).
            self._tool_acc += self._buf
            self._buf = ""
            tool_call = self._build_tool_call(self._tool_acc)
            self._tool_acc = ""
            if tool_call is not None:
                events.append(("tool", tool_call))
            return False

        if len(self._buf) > _HOLDBACK:
            cut = len(self._buf) - _HOLDBACK
            self._tool_acc += self._buf[:cut]
            self._buf = self._buf[cut:]
        return False

    def _flush_safe(self, events: List[ParserEvent], final: bool, *, kind: str) -> bool:
        """Flush as much of the current-mode buffer as is safe.

        Holds back the trailing ``_HOLDBACK`` chars unless ``final`` is True,
        because they might still grow into a tag.

        Returns ``False`` (break the drain loop) when we held back; ``True``
        only when we emptied the buffer (reaches the bottom of the while loop
        in ``_drain``).
        """
        if final:
            if self._buf:
                events.append((kind, self._buf))
                self._buf = ""
            return False
        if len(self._buf) > _HOLDBACK:
            cut = len(self._buf) - _HOLDBACK
            events.append((kind, self._buf[:cut]))
            self._buf = self._buf[cut:]
        return False

    # ------------------------------------------------------------------
    # Tool-call decoding
    # ------------------------------------------------------------------

    def _build_tool_call(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse the inside of a ``<tool_call>...</tool_call>`` block.

        Hermes models occasionally wrap the JSON in a fenced code block
        (\\`\\`\\`json ... \\`\\`\\`); strip those markers.
        """
        text = raw.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        try:
            data = json.loads(text)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(name, str) or not name:
            return None

        # OpenAI's tool_calls schema requires arguments to be a JSON-encoded
        # *string*. Ollama wants arguments as an *object*. We emit OpenAI shape
        # here; the Ollama wrapper re-decodes when building NDJSON.
        if isinstance(arguments, str):
            args_str = arguments
        else:
            try:
                args_str = json.dumps(arguments, ensure_ascii=False)
            except Exception:
                args_str = "{}"

        index = self._tool_index
        self._tool_index += 1
        tool_call: Dict[str, Any] = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "index": index,
            "function": {
                "name": name,
                "arguments": args_str,
            },
        }
        self.tool_calls.append(tool_call)
        return tool_call


def to_ollama_tool_call(openai_tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Translate an OpenAI-shaped tool_call into Ollama's NDJSON shape.

    Ollama omits ``id``/``type`` (they aren't part of the Ollama API contract)
    and stores ``arguments`` as a parsed object rather than a JSON string.
    """
    fn = openai_tool_call.get("function", {}) or {}
    args = fn.get("arguments", "{}")
    if isinstance(args, str):
        try:
            args_obj = json.loads(args)
        except Exception:
            args_obj = {}
    else:
        args_obj = args
    return {
        "function": {
            "name": fn.get("name", ""),
            "arguments": args_obj if isinstance(args_obj, (dict, list)) else {},
        }
    }


__all__ = [
    "THINK_OPEN",
    "THINK_CLOSE",
    "TOOL_OPEN",
    "TOOL_CLOSE",
    "ParserEvent",
    "StreamingParser",
    "parse_full_output",
    "to_ollama_tool_call",
]
