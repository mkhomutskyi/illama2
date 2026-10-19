"""Shared pytest fixtures + heavy-import stubs.

The runner imports ``openvino``, ``openvino_genai``, ``PIL``, ``numpy``, and
``requests`` at module load. CI does not have OpenVINO installed (and
should not need to, since these tests only exercise pure-Python helpers),
so we inject minimal stub modules into ``sys.modules`` BEFORE importing
``runner.server``. The stubs only need to satisfy attribute lookups that
happen at import time (mostly type annotations and class references in
function bodies). Anything heavier is patched per-test where needed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Make ``runner/`` importable as a top-level package, so ``import server``
# works from inside the tests just like the way uvicorn imports it inside
# the container at /app.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "runner"))


def _build_openvino_stub() -> types.ModuleType:
    mod = types.ModuleType("openvino")

    class _StubType:
        u8 = "u8"
        f32 = "f32"

    class _StubTensor:
        """Records the args it was constructed with so tests can inspect."""

        def __init__(self, dtype=None, shape=None, data=None):
            self.dtype = dtype
            self.shape = shape
            self.data = data

    mod.Type = _StubType
    mod.Tensor = _StubTensor
    return mod


def _build_openvino_genai_stub() -> types.ModuleType:
    mod = types.ModuleType("openvino_genai")

    class _StubGenerationConfig:
        def __init__(self):
            self.max_new_tokens = None
            self.temperature = None
            self.top_p = None
            self.do_sample = None
            self.stop_strings = None
            self.include_stop_str_in_output = None
            self.repetition_penalty = None
            self.rng_seed = None

    class _StubLLMPipeline:
        def __init__(self, *args, **kwargs):
            pass

    class _StubVLMPipeline:
        def __init__(self, *args, **kwargs):
            pass

    class _StreamingStatus:
        RUNNING = 0
        STOP = 1
        CANCEL = 2

    class _StubSchedulerConfig:
        """Mirrors the fields server.py touches on the real SchedulerConfig.

        cache_size defaults to 0 like the real one - that default is exactly
        why server.py always sets it explicitly.
        """

        def __init__(self):
            self.cache_size = 0
            self.enable_prefix_caching = False
            self.dynamic_split_fuse = False
            self.max_num_batched_tokens = 0
            self.max_num_seqs = 0
            self.num_linear_attention_blocks = 0

    def _stub_draft_model(models_path, device, **kwargs):
        return ("draft", str(models_path), device)

    mod.GenerationConfig = _StubGenerationConfig
    mod.LLMPipeline = _StubLLMPipeline
    mod.VLMPipeline = _StubVLMPipeline
    mod.StreamingStatus = _StreamingStatus
    mod.SchedulerConfig = _StubSchedulerConfig
    mod.draft_model = _stub_draft_model
    return mod


def _build_pil_stub() -> tuple[types.ModuleType, types.ModuleType]:
    pil = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")

    class _StubImage:
        def __init__(self, *args, **kwargs):
            self.size = (1, 1)
            self.mode = "RGB"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def convert(self, mode):
            self.mode = mode
            return self

    def _open(_buf):
        return _StubImage()

    image.open = _open
    pil.Image = image
    return pil, image


# Inject stubs ONLY if the real packages are not installed. Tests that
# import the real openvino still work locally on the dev host; CI gets the
# stubs and never touches the real Intel runtime.
def _maybe_install_stubs() -> None:
    try:
        import openvino  # noqa: F401
    except Exception:
        sys.modules["openvino"] = _build_openvino_stub()

    try:
        import openvino_genai  # noqa: F401
    except Exception:
        sys.modules["openvino_genai"] = _build_openvino_genai_stub()

    try:
        from PIL import Image  # noqa: F401
    except Exception:
        pil, image = _build_pil_stub()
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image

    try:
        import numpy  # noqa: F401
    except Exception:
        np = types.ModuleType("numpy")
        np.uint8 = "uint8"
        np.asarray = lambda *a, **kw: a[0] if a else None
        np.ascontiguousarray = lambda x: x
        sys.modules["numpy"] = np

    try:
        import requests  # noqa: F401
    except Exception:
        rq = types.ModuleType("requests")

        class _StubResp:
            headers: dict = {}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=0):
                return []

        def _get(*a, **kw):
            return _StubResp()

        rq.get = _get
        sys.modules["requests"] = rq


_maybe_install_stubs()
