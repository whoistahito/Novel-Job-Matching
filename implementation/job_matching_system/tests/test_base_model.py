def test_get_extractor_for_unknown_key_raises():
    from base_model import get_extractor_for

    try:
        get_extractor_for("does-not-exist")
    except ValueError as e:
        assert "Unknown" in str(e)
    else:
        raise AssertionError("Expected ValueError")


def test_get_extractor_for_caches_instances(monkeypatch):
    import base_model

    monkeypatch.setattr(base_model.LLMExtractor, "_load_model", lambda self: None)

    a = base_model.get_extractor_for("qwen3-8b")
    b = base_model.get_extractor_for("qwen3-8b")
    assert a is b


def test_process_text_merges_and_sorts(monkeypatch):
    import base_model
    from api_schema import Requirements

    monkeypatch.setattr(base_model.LLMExtractor, "_load_model", lambda self: None)
    monkeypatch.setattr(base_model, "chunk_markdown", lambda text, size: ["c1", "c2"])

    def fake_process_chunk(self, chunk):
        if chunk == "c1":
            return Requirements(skills=["b", "a"], experiences=["x"], qualifications=[])
        return Requirements(skills=["a"], experiences=[], qualifications=["q"])

    monkeypatch.setattr(base_model.LLMExtractor, "process_chunk", fake_process_chunk)

    ex = base_model.get_extractor_for("qwen3-8b")
    res = ex.process_text("ignored")

    assert res.skills == ["a", "b"]
    assert res.experiences == ["x"]
    assert res.qualifications == ["q"]
