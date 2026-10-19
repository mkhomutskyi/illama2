"""Unit tests for the tool_call + reasoning parser.

These deliberately skip the full server module - the parser has no
dependencies on OpenVINO, FastAPI, or any global state, so importing it
directly keeps the suite fast and reduces the blast radius of unrelated
runner changes.
"""

from __future__ import annotations

import json


import _parsers
from _parsers import StreamingParser, parse_full_output, to_ollama_tool_call


# ---------------------------------------------------------------------------
# parse_full_output (one-shot helper)
# ---------------------------------------------------------------------------


def test_plain_text_passes_through_unchanged():
    out = parse_full_output("Hello, world.")
    assert out == {"content": "Hello, world."}


def test_think_block_is_extracted_as_reasoning():
    out = parse_full_output("<think>Let me ponder.</think>The answer is 42.")
    assert out["content"] == "The answer is 42."
    assert out["reasoning_content"] == "Let me ponder."
    assert "tool_calls" not in out


def test_think_block_at_end_with_no_content():
    out = parse_full_output("<think>only thinking, no answer</think>")
    assert out["content"] == ""
    assert out["reasoning_content"] == "only thinking, no answer"


def test_tool_call_extracted_to_openai_shape():
    raw = (
        'I will check the weather. '
        '<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>'
    )
    out = parse_full_output(raw)
    # Pre-tool text becomes content; tool_call body is consumed.
    assert out["content"] == "I will check the weather. "
    assert "tool_calls" in out
    assert len(out["tool_calls"]) == 1
    tc = out["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    # arguments is an OpenAI-style JSON string.
    assert json.loads(tc["function"]["arguments"]) == {"location": "SF"}
    assert tc["index"] == 0
    assert tc["id"].startswith("call_")


def test_tool_call_with_string_arguments_passes_through():
    # Some Hermes-trained checkpoints emit arguments as a pre-encoded JSON string.
    raw = '<tool_call>{"name":"foo","arguments":"{\\"x\\":1}"}</tool_call>'
    out = parse_full_output(raw)
    assert out["tool_calls"][0]["function"]["arguments"] == '{"x":1}'


def test_tool_call_in_fenced_code_block():
    raw = (
        '<tool_call>```json\n{"name":"foo","arguments":{"a":1}}\n```</tool_call>'
    )
    out = parse_full_output(raw)
    assert out["tool_calls"][0]["function"]["name"] == "foo"


def test_two_tool_calls_get_distinct_indexes_and_ids():
    raw = (
        '<tool_call>{"name":"a","arguments":{}}</tool_call>'
        'between'
        '<tool_call>{"name":"b","arguments":{}}</tool_call>'
    )
    out = parse_full_output(raw)
    assert out["content"] == "between"
    indexes = [tc["index"] for tc in out["tool_calls"]]
    ids = [tc["id"] for tc in out["tool_calls"]]
    names = [tc["function"]["name"] for tc in out["tool_calls"]]
    assert indexes == [0, 1]
    assert names == ["a", "b"]
    assert len(set(ids)) == 2


def test_think_then_tool_call():
    raw = (
        '<think>I should call the tool.</think>'
        '<tool_call>{"name":"foo","arguments":{}}</tool_call>'
    )
    out = parse_full_output(raw)
    assert out["content"] == ""
    assert out["reasoning_content"] == "I should call the tool."
    assert len(out["tool_calls"]) == 1


def test_malformed_tool_call_is_dropped_silently():
    raw = '<tool_call>not json at all</tool_call>after'
    out = parse_full_output(raw)
    assert out["content"] == "after"
    assert "tool_calls" not in out


def test_tool_call_missing_name_is_dropped():
    raw = '<tool_call>{"arguments":{"x":1}}</tool_call>'
    out = parse_full_output(raw)
    assert "tool_calls" not in out


def test_unclosed_think_at_end_is_still_emitted():
    # If the model runs out of tokens mid-thought, surface what we got rather
    # than dropping it.
    out = parse_full_output("<think>partial reasoning")
    assert out.get("reasoning_content") == "partial reasoning"
    assert out["content"] == ""


def test_unclosed_tool_call_at_end_is_dropped_quietly():
    out = parse_full_output('<tool_call>{"name":"foo","argum')
    # No content, no tool call - we never close, so we have nothing to surface.
    assert out["content"] == ""
    assert "tool_calls" not in out


# ---------------------------------------------------------------------------
# StreamingParser (delta-by-delta behaviour)
# ---------------------------------------------------------------------------


def test_streaming_emits_content_immediately_when_no_tags():
    p = StreamingParser()
    # Need to push enough that holdback is exceeded.
    events = p.feed("Hello there, this is a long enough line.")
    kinds = [e[0] for e in events]
    assert all(k == "content" for k in kinds)
    flushed = "".join(e[1] for e in events)
    assert flushed in "Hello there, this is a long enough line."
    # Trailing characters held back; finalize must release them.
    final = p.finalize()
    full = flushed + "".join(e[1] for e in final)
    assert full == "Hello there, this is a long enough line."


def test_streaming_holds_back_potential_tag_prefix():
    p = StreamingParser()
    feed1 = "the answer is 42<thi"
    events = p.feed(feed1)
    flushed1 = "".join(e[1] for e in events if e[0] == "content")
    # The trailing potential-tag prefix must not be flushed yet.
    assert "<thi" not in flushed1
    # Some prefix of the input has been emitted; the rest is buffered until
    # we know it isn't a tag.
    assert feed1.startswith(flushed1)
    # Once disambiguating chars arrive, the held-back chunk is released as
    # plain content (it never was a real tag).
    events2 = p.feed(" was easy.")
    flushed_total = flushed1 + "".join(e[1] for e in events2 if e[0] == "content")
    flushed_total += "".join(e[1] for e in p.finalize() if e[0] == "content")
    assert flushed_total == feed1 + " was easy."


def test_streaming_think_split_across_chunks():
    """The closing tag may straddle two streamer callbacks."""
    p = StreamingParser()
    events: list = []
    for chunk in ["<think>hidd", "en thoughts</thi", "nk>visible"]:
        events.extend(p.feed(chunk))
    events.extend(p.finalize())

    reasoning = "".join(e[1] for e in events if e[0] == "reasoning")
    content = "".join(e[1] for e in events if e[0] == "content")
    assert reasoning == "hidden thoughts"
    assert content == "visible"


def test_streaming_tool_call_emits_atomically_on_close():
    p = StreamingParser()
    events: list = []
    chunks = [
        'pre',
        '<tool_call>',
        '{"name":"get_w',
        'eather","argum',
        'ents":{"loc":"SF"}}',
        '</tool_call>',
        'post',
    ]
    for c in chunks:
        events.extend(p.feed(c))
    events.extend(p.finalize())

    # We get content "pre", then exactly one tool event, then content "post".
    kinds = [e[0] for e in events]
    assert "tool" in kinds
    tool_events = [e for e in events if e[0] == "tool"]
    assert len(tool_events) == 1
    tc = tool_events[0][1]
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"loc": "SF"}

    content = "".join(e[1] for e in events if e[0] == "content")
    assert content == "prepost"


def test_streaming_no_emit_inside_tool_buffer():
    """While inside a tool_call we must not leak fragments to the caller."""
    p = StreamingParser()
    events_a = p.feed('<tool_call>{"name":"foo","arguments":{')
    events_b = p.feed('"x":1}')
    events_c = p.feed('}</tool_call>')
    # Only the closing chunk produces output.
    assert all(e[0] != "content" for e in events_a + events_b)
    assert all(e[0] != "reasoning" for e in events_a + events_b)
    final_tool = [e for e in events_c if e[0] == "tool"]
    assert len(final_tool) == 1
    assert final_tool[0][1]["function"]["name"] == "foo"


def test_streaming_holdback_does_not_split_tag():
    """Even single-character feeds must reconstruct <think> correctly."""
    p = StreamingParser()
    events: list = []
    for ch in "before<think>thoughts</think>after":
        events.extend(p.feed(ch))
    events.extend(p.finalize())

    content = "".join(e[1] for e in events if e[0] == "content")
    reasoning = "".join(e[1] for e in events if e[0] == "reasoning")
    assert content == "beforeafter"
    assert reasoning == "thoughts"


# ---------------------------------------------------------------------------
# to_ollama_tool_call (OpenAI -> Ollama shape)
# ---------------------------------------------------------------------------


def test_ollama_tool_call_converts_arguments_to_object():
    openai_tc = {
        "id": "call_x",
        "type": "function",
        "index": 0,
        "function": {"name": "foo", "arguments": '{"x":1}'},
    }
    ollama_tc = to_ollama_tool_call(openai_tc)
    assert "id" not in ollama_tc
    assert "type" not in ollama_tc
    assert ollama_tc["function"]["name"] == "foo"
    assert ollama_tc["function"]["arguments"] == {"x": 1}


def test_ollama_tool_call_handles_unparseable_arguments():
    bad = {"function": {"name": "foo", "arguments": "not-json"}}
    assert to_ollama_tool_call(bad)["function"]["arguments"] == {}


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exports_public_api():
    for name in ("StreamingParser", "parse_full_output", "to_ollama_tool_call"):
        assert hasattr(_parsers, name)


def test_holdback_is_at_least_longest_tag_minus_one():
    """Regression guard: shrinking _HOLDBACK below this would split tags."""
    longest = max(len(t) for t in (
        _parsers.THINK_OPEN,
        _parsers.THINK_CLOSE,
        _parsers.TOOL_OPEN,
        _parsers.TOOL_CLOSE,
    ))
    assert _parsers._HOLDBACK >= longest - 1
