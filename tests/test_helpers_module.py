"""Sanity tests that import directly from the extracted helper modules.

These are deliberately thin - the rich assertions live in
``test_server_helpers.py`` and exercise the same functions through
``server`` (which re-exports them). The point of this file is to prove the
split achieved its goal: ``_helpers`` and ``_schemas`` can be imported and
used WITHOUT touching the full server module (no OpenVINO, no pipeline, no
FastAPI app construction). A regression that re-introduces a hidden
dependency on server-level globals would fail an import here even if the
broader smoke suite still passes.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _helpers (pure utilities)
# ---------------------------------------------------------------------------


def test_helpers_imports_without_server():
    import _helpers

    assert callable(_helpers._b64decode_lenient)
    assert callable(_helpers.make_openai_error)
    assert callable(_helpers.is_auth_exempt_path)
    assert callable(_helpers.normalize_load_failure_policy)


def test_helpers_b64decode_lenient_round_trips():
    import _helpers

    encoded = "aGVsbG8sIHdvcmxk"  # "hello, world"
    assert _helpers._b64decode_lenient(encoded) == b"hello, world"


def test_helpers_data_url_re_extracts_payload():
    import _helpers

    m = _helpers._DATA_URL_RE.match("data:image/png;base64,AAAA")
    assert m is not None
    assert m.group("mime") == "image/png"
    assert m.group("payload") == "AAAA"


def test_helpers_auth_exempt_set_is_frozen():
    import _helpers

    assert isinstance(_helpers._AUTH_EXEMPT_PATHS, frozenset)
    assert "/health" in _helpers._AUTH_EXEMPT_PATHS
    assert "/v1/chat/completions" not in _helpers._AUTH_EXEMPT_PATHS
    with pytest.raises(AttributeError):
        _helpers._AUTH_EXEMPT_PATHS.add("/foo")  # type: ignore[attr-defined]


def test_helpers_make_openai_error_shape():
    import _helpers

    err = _helpers.make_openai_error("nope", error_type="invalid_request_error", param="model")
    assert err == {
        "message": "nope",
        "type": "invalid_request_error",
        "param": "model",
        "code": None,
    }


def test_helpers_make_stream_error_event_wraps_in_error_key():
    import _helpers

    evt = _helpers.make_stream_error_event("boom", code="generation_failed")
    assert "error" in evt
    assert evt["error"]["type"] == "server_error"
    assert evt["error"]["code"] == "generation_failed"


def test_helpers_build_usage_payload_computes_tps():
    import _helpers

    usage = _helpers.build_usage_payload(prompt_tokens=10, completion_tokens=20, generation_time=2.0)
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30
    assert usage["generation_time_seconds"] == 2.0
    assert usage["tokens_per_second"] == 10.0


def test_helpers_build_usage_payload_handles_zero_time():
    import _helpers

    usage = _helpers.build_usage_payload(prompt_tokens=5, completion_tokens=0, generation_time=0.0)
    assert usage["tokens_per_second"] == 0.0


def test_helpers_normalize_load_failure_policy_accepts_known_values():
    import _helpers

    assert _helpers.normalize_load_failure_policy("restart") == "restart"
    assert _helpers.normalize_load_failure_policy("RETURN_ERROR") == "return_error"
    assert _helpers.normalize_load_failure_policy("garbage") == "restart"


# ---------------------------------------------------------------------------
# _schemas (Pydantic models)
# ---------------------------------------------------------------------------


def test_schemas_imports_without_server():
    import _schemas

    assert hasattr(_schemas, "Message")
    assert hasattr(_schemas, "ChatCompletionRequest")
    assert hasattr(_schemas, "ImageURL")


def test_schemas_message_accepts_plain_string():
    from _schemas import Message

    msg = Message(role="user", content="hi")
    assert msg.role == "user"
    assert msg.content == "hi"


def test_schemas_message_accepts_multimodal_parts():
    from _schemas import ContentPartImage, ContentPartText, ImageURL, Message

    msg = Message(
        role="user",
        content=[
            ContentPartText(type="text", text="describe"),
            ContentPartImage(type="image_url", image_url=ImageURL(url="data:image/png;base64,AAAA")),
        ],
    )
    assert msg.role == "user"
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2


def test_schemas_chat_request_with_minimal_fields():
    from _schemas import ChatCompletionRequest, Message

    req = ChatCompletionRequest(
        model="qwen3-8b",
        messages=[Message(role="user", content="hi")],
    )
    assert req.model == "qwen3-8b"
    assert req.temperature == 0.7
    assert req.stream is False
