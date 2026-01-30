import pytest


def test_get_extractor_for_unknown_key_raises():
    from external_model import get_extractor_for

    with pytest.raises(ValueError):
        get_extractor_for("does-not-exist")


def test_get_extractor_for_caches_instances(monkeypatch):
    import external_model

    monkeypatch.setenv("EXTERNAL_LLM_API_KEY", "x")
    monkeypatch.setattr(external_model.ExternalLLMExtractor, "_load_model", lambda self: None)

    a = external_model.get_extractor_for("gpt-oss-120b")
    b = external_model.get_extractor_for("gpt-oss-120b")
    assert a is b


def test_missing_external_api_key_raises(monkeypatch):
    from external_model import ExternalLLMExtractor

    monkeypatch.delenv("EXTERNAL_LLM_API_KEY", raising=False)
    with pytest.raises(KeyError):
        ExternalLLMExtractor(model_name="m", chunk_size=10)


def test_process_text_merges_chunks(monkeypatch):
    import external_model
    from api_schema import Requirements

    monkeypatch.setenv("EXTERNAL_LLM_API_KEY", "x")
    monkeypatch.setattr(external_model.ExternalLLMExtractor, "_load_model", lambda self: None)
    monkeypatch.setattr(external_model, "chunk_markdown", lambda text, size: ["c1", "c2"])

    def fake_process_chunk(self, chunk):
        if chunk == "c1":
            return Requirements(skills=["b"], experiences=["e"], qualifications=[])
        return Requirements(skills=["a"], experiences=[], qualifications=["q"])

    monkeypatch.setattr(external_model.ExternalLLMExtractor, "process_chunk", fake_process_chunk)

    ex = external_model.get_extractor_for("gpt-oss-120b")
    res = ex.process_text("ignored")

    assert res.skills == ["a", "b"]
    assert res.experiences == ["e"]
    assert res.qualifications == ["q"]


def test_process_chunk_cleans_response(monkeypatch):
    import external_model

    monkeypatch.setenv("EXTERNAL_LLM_API_KEY", "x")
    monkeypatch.setattr(external_model.ExternalLLMExtractor, "_load_model", lambda self: None)

    ex = external_model.ExternalLLMExtractor(model_name="m", chunk_size=10)

    class DummyReq:
        pass

    captured = {}

    def fake_validate_json(s):
        captured["s"] = s
        return DummyReq()

    monkeypatch.setattr(external_model.Requirements, "model_validate_json", staticmethod(fake_validate_json))
    ex.generator = lambda *args, **kwargs: "```json\n{\"skills\":[]}\n```<|return|>\n"

    out = ex.process_chunk("chunk")
    assert isinstance(out, DummyReq)
    assert "\n" not in captured["s"]
    assert "```" not in captured["s"]
    assert "<|return|>" not in captured["s"]
