import sys
import types

import app as rag


def test_fastembed_uses_configured_cache(tmp_path, monkeypatch):
    calls = {}

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        types.SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )

    rag.FastEmbedAdapter("test/model", cache_dir=tmp_path / "models")

    assert calls == {
        "model_name": "test/model",
        "cache_dir": str(tmp_path / "models"),
    }
    assert (tmp_path / "models").is_dir()
