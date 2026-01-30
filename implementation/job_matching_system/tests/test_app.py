from fastapi.testclient import TestClient


def test_extract_endpoint_happy_path(monkeypatch):
    from inference import app as app_module
    from inference.api_schema import Requirements, SimilarityScore

    class StubExtractor:
        def process_text(self, text):
            return Requirements(skills=["python"], experiences=[], qualifications=[])

    monkeypatch.setattr(app_module, "get_extractor_for", lambda model_id: StubExtractor())
    monkeypatch.setattr(app_module, "compute_similarity", lambda user, req: SimilarityScore(score=0.5))

    client = TestClient(app_module.app)

    payload = {
        "modelId": "qwen3-8b",
        "inputText": "text",
        "userProfile": {"skills": ["python"], "experiences": [], "qualifications": []},
    }
    res = client.post("/extract", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["jobRequirements"]["skills"] == ["python"]
    assert body["similarityScore"]["score"] == 0.5


def test_extract_endpoint_returns_500_on_exception(monkeypatch):
    from inference import app as app_module

    def boom(_):
        raise RuntimeError("bad")

    monkeypatch.setattr(app_module, "get_extractor_for", boom)

    client = TestClient(app_module.app)
    payload = {
        "modelId": "qwen3-8b",
        "inputText": "text",
        "userProfile": {"skills": [], "experiences": [], "qualifications": []},
    }
    res = client.post("/extract", json=payload)
    assert res.status_code == 500
    assert "bad" in res.json()["detail"]
