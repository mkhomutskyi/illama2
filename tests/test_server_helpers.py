"""Smoke tests for the pure-Python helpers in runner/server.py.

These tests deliberately avoid anything that needs a real OpenVINO
runtime, GPU, or model on disk. They cover the helpers that have been
the source of recent regressions:

  - SSRF guard for VLM image_url fetches (_assert_image_url_is_safe)
  - whitespace-tolerant base64 decoding (_b64decode_lenient)
  - data-URL regex (_DATA_URL_RE)
  - multimodal content flattening (_message_text)
  - config parsing helpers (_get_config_bool, _get_config_int,
    _parse_idle_timeout)
  - auth-exempt path set (is_auth_exempt_path)
  - OpenAI-shaped error builder (make_openai_error)
  - version resolution (_read_illama_version)
"""

from __future__ import annotations

import base64
from unittest import mock

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# _clean_openvino_error / _is_user_load_error
# ---------------------------------------------------------------------------


class TestCleanOpenVinoError:
    def test_strips_jenkins_build_path_prefix(self):
        import server

        text = (
            "Exception from /home/jenkins/agent/workspace/private-ci/ie/build-linux"
            "-manylinux_2_28/b/repos/openvino.genai/src/cpp/src/visual_language/"
            "vlm_config.cpp:34:\nUnsupported 'gemma4' VLM model type\n"
        )
        cleaned = server._clean_openvino_error(Exception(text))
        assert cleaned == "Unsupported 'gemma4' VLM model type"

    def test_keeps_first_line_when_no_jenkins_prefix(self):
        import server

        cleaned = server._clean_openvino_error(RuntimeError("Something blew up"))
        assert cleaned == "Something blew up"

    def test_handles_empty_exception_text(self):
        import server

        cleaned = server._clean_openvino_error(Exception(""))
        assert cleaned == "Exception"


class TestIsUserLoadError:
    def test_unsupported_vlm_model_type_is_user_error(self):
        import server

        exc = Exception(
            "Exception from /tmp/vlm_config.cpp:34:\nUnsupported 'gemma4' VLM model type"
        )
        assert server._is_user_load_error(exc) is True

    def test_unsupported_llm_model_type_is_user_error(self):
        import server

        exc = Exception("Unsupported 'futuremodel' LLM model type")
        assert server._is_user_load_error(exc) is True

    def test_missing_tokenizer_is_user_error(self):
        import server

        exc = Exception("Failed to open tokenizer.xml")
        assert server._is_user_load_error(exc) is True

    def test_gpu_oom_is_NOT_user_error(self):
        import server

        exc = Exception("CL_OUT_OF_RESOURCES at level zero")
        assert server._is_user_load_error(exc) is False

    def test_generic_runtime_error_is_NOT_user_error(self):
        import server

        exc = Exception("Tensor shape mismatch [1,3,224,224] vs [1,3,448,448]")
        assert server._is_user_load_error(exc) is False


# Imported via conftest's sys.path tweak; conftest also stubs the heavy
# modules so this import works without OpenVINO/PIL/numpy/requests
# installed.
import server  # type: ignore[import-not-found]  # noqa: E402 - after stub setup


# ---------------------------------------------------------------------------
# _b64decode_lenient
# ---------------------------------------------------------------------------


class TestB64DecodeLenient:
    def test_plain_base64_round_trips(self):
        original = b"hello, world"
        encoded = base64.b64encode(original).decode()
        assert server._b64decode_lenient(encoded) == original

    def test_ignores_embedded_newlines(self):
        original = b"x" * 200
        wrapped = "\n".join(
            base64.b64encode(original).decode()[i : i + 76]
            for i in range(0, len(base64.b64encode(original).decode()), 76)
        )
        assert "\n" in wrapped
        assert server._b64decode_lenient(wrapped) == original

    def test_ignores_tabs_and_spaces(self):
        encoded = base64.b64encode(b"abc").decode()
        spaced = encoded[0] + " " + encoded[1] + "\t" + encoded[2:]
        assert server._b64decode_lenient(spaced) == b"abc"

    def test_rejects_garbage(self):
        with pytest.raises(Exception):
            server._b64decode_lenient("not!valid$base64")


# ---------------------------------------------------------------------------
# _DATA_URL_RE
# ---------------------------------------------------------------------------


class TestDataUrlRegex:
    def test_matches_image_png_data_url(self):
        m = server._DATA_URL_RE.match("data:image/png;base64,AAAA")
        assert m is not None
        assert m.group("mime") == "image/png"
        assert m.group("payload") == "AAAA"

    def test_matches_image_jpeg_data_url(self):
        m = server._DATA_URL_RE.match("data:image/jpeg;base64,zzzz==")
        assert m is not None
        assert m.group("mime") == "image/jpeg"

    def test_does_not_match_http_url(self):
        assert server._DATA_URL_RE.match("https://example.com/cat.png") is None

    def test_does_not_match_raw_base64(self):
        assert server._DATA_URL_RE.match("AAAAAAAA") is None


# ---------------------------------------------------------------------------
# _assert_image_url_is_safe (SSRF guard)
# ---------------------------------------------------------------------------


def _addrinfo_for(ip: str):
    """Shape that ``socket.getaddrinfo`` returns. We only need entry[4][0]."""
    return [(0, 0, 0, "", (ip, 0))]


class TestAssertImageUrlIsSafe:
    def test_blocks_loopback(self, monkeypatch):
        monkeypatch.setattr(server, "ALLOW_PRIVATE_IMAGE_URLS", False)
        monkeypatch.setattr(
            server.socket, "getaddrinfo", lambda host, _port: _addrinfo_for("127.0.0.1")
        )
        with pytest.raises(HTTPException) as excinfo:
            server._assert_image_url_is_safe("http://localhost/cat.png")
        assert excinfo.value.status_code == 400
        assert "Private/loopback" in str(excinfo.value.detail)

    def test_blocks_link_local_metadata_service(self, monkeypatch):
        monkeypatch.setattr(server, "ALLOW_PRIVATE_IMAGE_URLS", False)
        monkeypatch.setattr(
            server.socket,
            "getaddrinfo",
            lambda host, _port: _addrinfo_for("169.254.169.254"),
        )
        with pytest.raises(HTTPException):
            server._assert_image_url_is_safe("http://metadata/cat.png")

    @pytest.mark.parametrize(
        "ip",
        ["10.0.0.1", "172.16.0.5", "192.168.1.1"],
    )
    def test_blocks_rfc1918_private_ranges(self, monkeypatch, ip):
        monkeypatch.setattr(server, "ALLOW_PRIVATE_IMAGE_URLS", False)
        monkeypatch.setattr(
            server.socket, "getaddrinfo", lambda host, _port: _addrinfo_for(ip)
        )
        with pytest.raises(HTTPException):
            server._assert_image_url_is_safe(f"http://internal/{ip}.png")

    def test_blocks_ipv6_loopback(self, monkeypatch):
        monkeypatch.setattr(server, "ALLOW_PRIVATE_IMAGE_URLS", False)
        monkeypatch.setattr(
            server.socket, "getaddrinfo", lambda host, _port: _addrinfo_for("::1")
        )
        with pytest.raises(HTTPException):
            server._assert_image_url_is_safe("http://[::1]/cat.png")

    def test_allows_public_ip(self, monkeypatch):
        monkeypatch.setattr(server, "ALLOW_PRIVATE_IMAGE_URLS", False)
        monkeypatch.setattr(
            server.socket, "getaddrinfo", lambda host, _port: _addrinfo_for("1.1.1.1")
        )
        # Should NOT raise.
        server._assert_image_url_is_safe("https://one.one.one.one/cat.png")

    def test_blocks_when_any_address_is_private(self, monkeypatch):
        # A hostname returning both a public IP and 127.0.0.1 must be
        # rejected so requests cannot pick the loopback at connect time.
        monkeypatch.setattr(server, "ALLOW_PRIVATE_IMAGE_URLS", False)
        monkeypatch.setattr(
            server.socket,
            "getaddrinfo",
            lambda host, _port: [
                (0, 0, 0, "", ("8.8.8.8", 0)),
                (0, 0, 0, "", ("127.0.0.1", 0)),
            ],
        )
        with pytest.raises(HTTPException):
            server._assert_image_url_is_safe("http://mixed.example.com/cat.png")

    def test_override_via_env_flag(self, monkeypatch):
        monkeypatch.setattr(server, "ALLOW_PRIVATE_IMAGE_URLS", True)
        # Even though we'd resolve to loopback, the override skips the check.
        server._assert_image_url_is_safe("http://localhost/cat.png")

    def test_rejects_url_without_host(self, monkeypatch):
        monkeypatch.setattr(server, "ALLOW_PRIVATE_IMAGE_URLS", False)
        with pytest.raises(HTTPException) as excinfo:
            server._assert_image_url_is_safe("file:///etc/passwd")
        assert excinfo.value.status_code == 400


# ---------------------------------------------------------------------------
# _message_text (multimodal content flattening)
# ---------------------------------------------------------------------------


class TestMessageText:
    def test_plain_string_content_is_returned_unchanged(self):
        msg = server.Message(role="user", content="hello")
        assert server._message_text(msg) == "hello"

    def test_multimodal_text_parts_are_joined_with_newline(self):
        # _message_text uses "\n".join so distinct text parts stay
        # logically separated for chat-template builders downstream.
        msg = server.Message(
            role="user",
            content=[
                {"type": "text", "text": "Hello, "},
                {"type": "text", "text": "world!"},
            ],
        )
        assert server._message_text(msg) == "Hello, \nworld!"

    def test_image_parts_are_dropped(self):
        msg = server.Message(
            role="user",
            content=[
                {"type": "text", "text": "Describe this:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        )
        assert server._message_text(msg) == "Describe this:"

    def test_only_image_parts_yields_empty_string(self):
        msg = server.Message(
            role="user",
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cat.png"},
                },
            ],
        )
        assert server._message_text(msg) == ""


# ---------------------------------------------------------------------------
# Config parsing helpers
# ---------------------------------------------------------------------------


class TestGetConfigBool:
    @pytest.mark.parametrize("value", [True, "true", "1", "yes", "on", "TRUE"])
    def test_truthy_values(self, value):
        assert server._get_config_bool({"k": value}, "k", default=False) is True

    @pytest.mark.parametrize("value", [False, "false", "0", "no", "off", "FALSE"])
    def test_falsy_values(self, value):
        assert server._get_config_bool({"k": value}, "k", default=True) is False

    def test_missing_key_returns_default(self):
        assert server._get_config_bool({}, "k", default=True) is True

    def test_invalid_value_returns_default(self):
        assert server._get_config_bool({"k": "maybe"}, "k", default=True) is True


class TestParseIdleTimeout:
    def test_seconds_default(self):
        assert server._parse_idle_timeout({"idle_timeout": "60"}, default_seconds=10) == 60

    def test_seconds_suffix(self):
        assert server._parse_idle_timeout({"idle_timeout": "30s"}, default_seconds=0) == 30

    def test_minutes_suffix(self):
        assert server._parse_idle_timeout({"idle_timeout": "5m"}, default_seconds=0) == 300

    def test_unsupported_suffix_falls_back_to_default(self):
        # Only s and m are supported; an h suffix should be rejected and
        # the default returned. This is a regression-protection assertion
        # against accidentally accepting (but mis-interpreting) "2h".
        assert (
            server._parse_idle_timeout({"idle_timeout": "2h"}, default_seconds=42)
            == 42
        )

    def test_zero_disables(self):
        assert server._parse_idle_timeout({"idle_timeout": "0"}, default_seconds=60) == 0

    def test_invalid_falls_back_to_default(self):
        assert (
            server._parse_idle_timeout({"idle_timeout": "garbage"}, default_seconds=42)
            == 42
        )


# ---------------------------------------------------------------------------
# _watchdog_sleep_interval
# ---------------------------------------------------------------------------


class TestWatchdogSleepInterval:
    """Drives _watchdog_sleep_interval with a representative range so the
    'check 4x per window, floored at 1, capped at 30, 60 when disabled'
    contract is locked in."""

    @pytest.mark.parametrize(
        "idle,expected",
        [
            (0, 60),     # disabled -> watchdog never starts; interval is moot
            (-5, 60),    # negative -> same as disabled
            (1, 1),      # tiny timeout -> floor at 1s (no busy-loop)
            (4, 1),      # 4//4 = 1
            (15, 3),     # short timeout: 15//4 = 3, fires within ~18s
            (60, 15),    # current default: 60//4 = 15, parity with the
                         # original hard-coded sleep(60) behaviour
            (120, 30),   # cap kicks in at 30s
            (3600, 30),  # 1h timeout still wakes only twice per minute
        ],
    )
    def test_buckets(self, idle, expected):
        assert server._watchdog_sleep_interval(idle) == expected


# ---------------------------------------------------------------------------
# _ollama_timestamp (now optionally takes a unix epoch)
# ---------------------------------------------------------------------------


class TestOllamaTimestamp:
    def test_default_uses_now(self):
        # No argument -> formats the current UTC second; we just verify the
        # shape ("YYYY-MM-DDTHH:MM:SS.000Z") rather than the exact value.
        out = server._ollama_timestamp()
        assert len(out) == len("2026-05-03T20:30:00.000Z")
        assert out.endswith(".000Z")
        assert out[4] == "-" and out[10] == "T" and out[13] == ":"

    def test_explicit_epoch_is_formatted_in_utc(self):
        # 1700000000 is 2023-11-14T22:13:20Z; verifies we treat the input
        # as a unix timestamp and not as anything timezone-relative.
        assert (
            server._ollama_timestamp(1700000000)
            == "2023-11-14T22:13:20.000Z"
        )

    def test_negative_input_clamped_to_epoch(self):
        # Anything < 0 (e.g. last_request_time uninitialised) clamps to 0
        # so we never emit a 1969 timestamp.
        assert server._ollama_timestamp(-1).startswith("1970-01-01T00:00:00")

    def test_never_expires_sentinel_matches_ollama(self):
        # Stock Ollama emits "9999-12-31T23:59:59.999999999Z" for
        # keep_alive=-1; our sentinel uses the simpler millis format the
        # rest of /api/* uses. This test pins the exact string so a future
        # accidental rewrite doesn't break dashboard clients that match
        # on the year prefix.
        assert server._OLLAMA_NEVER_EXPIRES == "9999-12-31T23:59:59.000Z"


# ---------------------------------------------------------------------------
# is_auth_exempt_path
# ---------------------------------------------------------------------------


class TestAuthExemptPath:
    @pytest.mark.parametrize(
        "path",
        ["/", "/health", "/openapi.json", "/docs", "/redoc", "/api/version"],
    )
    def test_documented_paths_are_exempt(self, path):
        assert server.is_auth_exempt_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/models",
            "/api/chat",
            "/api/generate",
            "/illama/health",
            "/metrics",
        ],
    )
    def test_protected_paths_are_not_exempt(self, path):
        assert server.is_auth_exempt_path(path) is False


# ---------------------------------------------------------------------------
# make_openai_error
# ---------------------------------------------------------------------------


class TestMakeOpenAIError:
    def test_minimal_error_shape(self):
        err = server.make_openai_error("oops")
        assert err["message"] == "oops"
        assert err["type"] == "invalid_request_error"
        assert err.get("param") is None
        assert err.get("code") is None

    def test_includes_optional_fields(self):
        err = server.make_openai_error(
            "bad temperature",
            error_type="invalid_request_error",
            param="temperature",
            code="param_out_of_range",
        )
        assert err["param"] == "temperature"
        assert err["code"] == "param_out_of_range"


# ---------------------------------------------------------------------------
# _read_illama_version
# ---------------------------------------------------------------------------


class TestReadIllamaVersion:
    def test_env_value_wins(self, monkeypatch):
        monkeypatch.setenv("ILLAMA_VERSION", "9.9.9")
        assert server._read_illama_version() == "9.9.9"

    def test_falls_back_to_constant_when_no_env_no_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ILLAMA_VERSION", raising=False)
        # Force the version-file lookup to a non-existent path so we hit
        # the hardcoded fallback.
        with mock.patch("server.os.path.exists", return_value=False):
            assert server._read_illama_version() == server._VERSION_FALLBACK

    def test_reads_from_file_when_env_absent(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ILLAMA_VERSION", raising=False)
        version_file = tmp_path / "VERSION"
        version_file.write_text("3.4.5\n")
        # Patch the os.path.dirname result so _read_illama_version finds OUR
        # file. Use a small builder to keep the patch scope tight.
        with mock.patch("server.os.path.abspath", return_value=str(tmp_path / "server.py")):
            with mock.patch("server.os.path.dirname", return_value=str(tmp_path)):
                with mock.patch("server.os.path.exists", side_effect=lambda p: True):
                    assert server._read_illama_version() == "3.4.5"


# ---------------------------------------------------------------------------
# 2.5.0 - pipeline backend selection + per-model overrides
# ---------------------------------------------------------------------------


def test_normalize_pipeline_backend_accepts_known_values():
    import server

    assert server.normalize_pipeline_backend("auto") == "auto"
    assert server.normalize_pipeline_backend("STATEFUL") == "stateful"
    assert server.normalize_pipeline_backend(" continuous ") == "continuous"


def test_normalize_pipeline_backend_falls_back_to_auto():
    import server

    assert server.normalize_pipeline_backend("garbage") == "auto"
    assert server.normalize_pipeline_backend("") == "auto"


def test_coerce_bool_parses_json_and_string_forms():
    import server

    assert server._coerce_bool(True) is True
    assert server._coerce_bool("yes") is True
    assert server._coerce_bool("OFF") is False
    assert server._coerce_bool(0) is None   # ints are not accepted
    assert server._coerce_bool("maybe") is None


def test_new_override_keys_are_registered():
    import server

    for key in ("prompt_lookup", "draft_model", "num_assistant_tokens",
                "scheduler_cache_size_gb"):
        assert key in server.MODEL_OVERRIDE_KEYS


def test_get_model_override_bool_uses_default_when_unset(monkeypatch):
    import server

    monkeypatch.setattr(server, "get_model_overrides", lambda name: {})
    assert server._get_model_override_bool("m", "prompt_lookup", True) is True
    assert server._get_model_override_bool("m", "prompt_lookup", False) is False


def test_get_model_override_bool_reads_override(monkeypatch):
    import server

    monkeypatch.setattr(server, "get_model_overrides", lambda name: {"prompt_lookup": "on"})
    assert server._get_model_override_bool("m", "prompt_lookup", False) is True


def test_get_model_override_bool_rejects_unregistered_key(monkeypatch):
    import server

    monkeypatch.setattr(server, "get_model_overrides", lambda name: {"bogus": True})
    # Unregistered keys must not be honoured, even if present in model.json.
    assert server._get_model_override_bool("m", "bogus", False) is False


def test_resolve_draft_model_dir_returns_none_when_unset(monkeypatch):
    import server

    monkeypatch.setattr(server, "get_model_overrides", lambda name: {})
    assert server._resolve_draft_model_dir("target") is None


def test_resolve_draft_model_dir_returns_none_for_missing_dir(monkeypatch, tmp_path):
    import server

    monkeypatch.setattr(server, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(server, "get_model_overrides", lambda name: {"draft_model": "nope"})
    assert server._resolve_draft_model_dir("target") is None


def test_resolve_draft_model_dir_requires_openvino_xml(monkeypatch, tmp_path):
    import server

    draft = tmp_path / "drafter"
    draft.mkdir()
    monkeypatch.setattr(server, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(server, "get_model_overrides", lambda name: {"draft_model": "drafter"})
    # Directory exists but has no IR -> rejected.
    assert server._resolve_draft_model_dir("target") is None

    (draft / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    assert server._resolve_draft_model_dir("target") == str(draft)


def test_resolve_draft_model_dir_rejects_vlm_shaped_drafter(monkeypatch, tmp_path):
    """ov_genai.draft_model() needs a text-only IR (openvino_model.xml).

    A VLM-shaped export otherwise fails deep in the OpenVINO frontend with an
    opaque file-not-found error, so it must be rejected up front.
    """
    import server

    draft = tmp_path / "vlm-drafter"
    draft.mkdir()
    (draft / "openvino_language_model.xml").write_text("<net/>", encoding="utf-8")
    monkeypatch.setattr(server, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(
        server, "get_model_overrides", lambda name: {"draft_model": "vlm-drafter"}
    )
    assert server._resolve_draft_model_dir("target") is None


def test_resolve_draft_model_dir_strips_path_components(monkeypatch, tmp_path):
    """A draft_model override names a registry entry, not an arbitrary path."""
    import server

    draft = tmp_path / "drafter"
    draft.mkdir()
    (draft / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    monkeypatch.setattr(server, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(
        server, "get_model_overrides", lambda name: {"draft_model": "../../etc/drafter"}
    )
    # basename() collapses the traversal attempt back into the registry.
    assert server._resolve_draft_model_dir("target") == str(draft)


def test_cache_size_ladder_doubles_up_to_max():
    import server

    assert server._cache_size_ladder(2, 8) == [2, 4, 8]
    assert server._cache_size_ladder(1, 8) == [1, 2, 4, 8]


def test_cache_size_ladder_handles_start_at_or_above_max():
    import server

    assert server._cache_size_ladder(8, 8) == [8]
    assert server._cache_size_ladder(16, 8) == [8]


def test_cache_size_ladder_empty_when_max_is_zero():
    import server

    assert server._cache_size_ladder(2, 0) == []


def test_cache_size_ladder_is_ascending_and_deduped():
    import server

    ladder = server._cache_size_ladder(3, 12)
    assert ladder == sorted(set(ladder))
    assert ladder[-1] == 12


def test_is_cache_too_small_error_matches_allocator_message():
    import server

    exc = RuntimeError(
        "Requested linear attention cache allocation exceeds the configured cache size."
    )
    assert server._is_cache_too_small_error(exc) is True


def test_is_cache_too_small_error_ignores_unrelated_failures():
    import server

    assert server._is_cache_too_small_error(RuntimeError("Unsupported 'qwen9' model type")) is False
    assert server._is_cache_too_small_error(RuntimeError("CL_OUT_OF_RESOURCES")) is False


def test_build_scheduler_config_sets_cache_and_prefix(monkeypatch):
    import server

    sched = server._build_scheduler_config(4)
    assert sched is not None
    assert sched.cache_size == 4
    assert sched.enable_prefix_caching == server.ENABLE_PREFIX_CACHING


def test_build_scheduler_config_returns_none_without_runtime_support(monkeypatch):
    import server

    monkeypatch.delattr(server.ov_genai, "SchedulerConfig", raising=False)
    assert server._build_scheduler_config(4) is None


def test_build_pipeline_escalates_cache_until_allocator_accepts(monkeypatch):
    """Mirrors the real Qwen3.5 behaviour: refuses 2/4 GB, constructs at 8."""
    import server

    attempts = []

    class _Pipe:
        pass

    def _fake_llm(models_path, device, **kwargs):
        gb = kwargs["scheduler_config"].cache_size
        attempts.append(gb)
        if gb < 8:
            raise RuntimeError(
                "Requested linear attention cache allocation exceeds the configured cache size."
            )
        return _Pipe()

    monkeypatch.setattr(server.ov_genai, "LLMPipeline", _fake_llm)
    monkeypatch.setattr(server, "PIPELINE_BACKEND", "auto")
    monkeypatch.setattr(server, "SCHEDULER_CACHE_SIZE_GB", 2)
    monkeypatch.setattr(server, "SCHEDULER_CACHE_MAX_GB", 8)
    monkeypatch.setattr(server, "get_model_overrides", lambda name: {})

    pipeline, backend = server._build_pipeline("/models/m", "m", is_vlm=False)
    assert backend == "continuous"
    assert isinstance(pipeline, _Pipe)
    assert attempts == [2, 4, 8]


def test_build_pipeline_falls_back_to_stateful_on_real_error(monkeypatch):
    """A non-cache error must not be retried, and must degrade gracefully."""
    import server

    calls = {"continuous": 0, "stateful": 0}

    def _fake_llm(models_path, device, **kwargs):
        if "scheduler_config" in kwargs:
            calls["continuous"] += 1
            raise RuntimeError("Unsupported 'qwen9_9' model type")
        calls["stateful"] += 1
        return object()

    monkeypatch.setattr(server.ov_genai, "LLMPipeline", _fake_llm)
    monkeypatch.setattr(server, "PIPELINE_BACKEND", "auto")
    monkeypatch.setattr(server, "SCHEDULER_CACHE_SIZE_GB", 2)
    monkeypatch.setattr(server, "SCHEDULER_CACHE_MAX_GB", 8)
    monkeypatch.setattr(server, "get_model_overrides", lambda name: {})

    _pipeline, backend = server._build_pipeline("/models/m", "m", is_vlm=False)
    assert backend == "stateful"
    # Tried continuous exactly once (no retry on a non-cache error).
    assert calls == {"continuous": 1, "stateful": 1}


def test_build_pipeline_continuous_mode_raises_instead_of_degrading(monkeypatch):
    import server

    def _fake_llm(models_path, device, **kwargs):
        raise RuntimeError("Unsupported 'qwen9_9' model type")

    monkeypatch.setattr(server.ov_genai, "LLMPipeline", _fake_llm)
    monkeypatch.setattr(server, "PIPELINE_BACKEND", "continuous")
    monkeypatch.setattr(server, "SCHEDULER_CACHE_SIZE_GB", 2)
    monkeypatch.setattr(server, "SCHEDULER_CACHE_MAX_GB", 8)
    monkeypatch.setattr(server, "get_model_overrides", lambda name: {})

    with pytest.raises(RuntimeError):
        server._build_pipeline("/models/m", "m", is_vlm=False)


def test_build_pipeline_vlm_passes_scheduler_config_as_kwarg(monkeypatch):
    """VLMPipeline has no config mapping - it must arrive as a keyword."""
    import server

    seen = {}

    def _fake_vlm(models_path, device, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(server.ov_genai, "VLMPipeline", _fake_vlm)
    monkeypatch.setattr(server, "PIPELINE_BACKEND", "auto")
    monkeypatch.setattr(server, "SCHEDULER_CACHE_SIZE_GB", 8)
    monkeypatch.setattr(server, "SCHEDULER_CACHE_MAX_GB", 8)
    monkeypatch.setattr(server, "get_model_overrides", lambda name: {})

    _pipeline, backend = server._build_pipeline("/models/m", "m", is_vlm=True)
    assert backend == "continuous"
    assert "scheduler_config" in seen
    assert seen["scheduler_config"].cache_size == 8


def test_build_pipeline_stateful_mode_skips_scheduler(monkeypatch):
    import server

    seen = {}

    def _fake_llm(models_path, device, **kwargs):
        seen["config"] = kwargs.get("scheduler_config")
        return object()

    monkeypatch.setattr(server.ov_genai, "LLMPipeline", _fake_llm)
    monkeypatch.setattr(server, "PIPELINE_BACKEND", "stateful")
    monkeypatch.setattr(server, "get_model_overrides", lambda name: {})

    _pipeline, backend = server._build_pipeline("/models/m", "m", is_vlm=False)
    assert backend == "stateful"
    assert seen["config"] is None


def test_scheduler_config_disables_prefix_caching_for_draft_models(monkeypatch):
    """Prefix caching + speculative decoding trips a sequence_group assertion."""
    import server

    monkeypatch.setattr(server, "ENABLE_PREFIX_CACHING", True)
    with_draft = server._build_scheduler_config(2, allow_prefix_caching=False)
    without_draft = server._build_scheduler_config(2, allow_prefix_caching=True)
    assert with_draft.enable_prefix_caching is False
    assert without_draft.enable_prefix_caching is True


def test_scheduler_config_respects_global_prefix_caching_off(monkeypatch):
    import server

    monkeypatch.setattr(server, "ENABLE_PREFIX_CACHING", False)
    sched = server._build_scheduler_config(2, allow_prefix_caching=True)
    assert sched.enable_prefix_caching is False


def test_build_pipeline_turns_off_prefix_caching_when_draft_linked(monkeypatch, tmp_path):
    import server

    draft = tmp_path / "drafter"
    draft.mkdir()
    (draft / "openvino_model.xml").write_text("<net/>", encoding="utf-8")

    seen = {}

    def _fake_llm(models_path, device, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(server.ov_genai, "LLMPipeline", _fake_llm)
    monkeypatch.setattr(server, "PIPELINE_BACKEND", "auto")
    monkeypatch.setattr(server, "ENABLE_PREFIX_CACHING", True)
    monkeypatch.setattr(server, "SCHEDULER_CACHE_SIZE_GB", 2)
    monkeypatch.setattr(server, "SCHEDULER_CACHE_MAX_GB", 2)
    monkeypatch.setattr(server, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(
        server, "get_model_overrides", lambda name: {"draft_model": "drafter"}
    )

    _pipeline, backend = server._build_pipeline("/models/m", "m", is_vlm=False)
    assert backend == "continuous"
    assert "draft_model" in seen
    assert seen["scheduler_config"].enable_prefix_caching is False


def test_build_pipeline_passes_scheduler_config_as_kwarg_not_positional(monkeypatch):
    """The positional `config` mapping is deprecated in openvino-genai 2026.4."""
    import server

    captured = {}

    def _fake_llm(models_path, device, *args, **kwargs):
        captured["positional"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(server.ov_genai, "LLMPipeline", _fake_llm)
    monkeypatch.setattr(server, "PIPELINE_BACKEND", "auto")
    monkeypatch.setattr(server, "SCHEDULER_CACHE_SIZE_GB", 2)
    monkeypatch.setattr(server, "SCHEDULER_CACHE_MAX_GB", 2)
    monkeypatch.setattr(server, "get_model_overrides", lambda name: {})

    server._build_pipeline("/models/m", "m", is_vlm=False)
    assert captured["positional"] == ()
    assert "scheduler_config" in captured["kwargs"]


# ---------------------------------------------------------------------------
# 2.6.0 - drafter-as-alias resolution
# ---------------------------------------------------------------------------


def test_resolve_model_alias_passes_through_ordinary_models(monkeypatch):
    import server

    monkeypatch.setattr(server, "load_model_metadata", lambda name: {})
    assert server.resolve_model_alias("plain") == ("plain", None)


def test_resolve_model_alias_maps_drafter_to_base(monkeypatch, tmp_path):
    import server

    (tmp_path / "base").mkdir()
    monkeypatch.setattr(server, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(
        server, "load_model_metadata",
        lambda name: {"role": "draft", "base_model": "base"},
    )
    assert server.resolve_model_alias("drafter") == ("base", "drafter")


def test_resolve_model_alias_ignores_missing_base(monkeypatch, tmp_path):
    import server

    monkeypatch.setattr(server, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(
        server, "load_model_metadata",
        lambda name: {"role": "draft", "base_model": "gone"},
    )
    # Falls back to loading the drafter itself rather than erroring.
    assert server.resolve_model_alias("drafter") == ("drafter", None)


def test_resolve_model_alias_requires_role_draft(monkeypatch, tmp_path):
    import server

    (tmp_path / "base").mkdir()
    monkeypatch.setattr(server, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(
        server, "load_model_metadata", lambda name: {"base_model": "base"}
    )
    assert server.resolve_model_alias("x") == ("x", None)


def test_resolve_model_alias_strips_path_traversal(monkeypatch, tmp_path):
    import server

    (tmp_path / "base").mkdir()
    monkeypatch.setattr(server, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(
        server, "load_model_metadata",
        lambda name: {"role": "draft", "base_model": "../../etc/base"},
    )
    assert server.resolve_model_alias("drafter") == ("base", "drafter")


def test_build_acceleration_kwargs_honours_forced_draft(monkeypatch, tmp_path):
    import server

    draft = tmp_path / "drafter"
    draft.mkdir()
    (draft / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    monkeypatch.setattr(server, "MODELS_DIR", str(tmp_path))
    # No overrides on the target - the draft comes purely from forced_draft.
    monkeypatch.setattr(server, "get_model_overrides", lambda name: {})
    monkeypatch.setattr(server, "PROMPT_LOOKUP_DEFAULT", False)

    extra = server._build_acceleration_kwargs("base", forced_draft="drafter")
    assert "draft_model" in extra
