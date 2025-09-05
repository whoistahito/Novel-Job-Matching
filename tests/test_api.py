from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_models_list_includes_echo_and_all_adapters():
    # Trigger startup event to register adapters
    with TestClient(app) as c:
        resp = c.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        models = set(data.get("models", []))
        expected = {
            "echo",
            "glm4-9b",
            "glm4-z1-9b",
            "llama3.1-nemotron-8b",
            "mistral-small-24b",
            "qwen3-8b",
        }
        assert expected.issubset(models)


def test_inference_echo_happy_path():
    with TestClient(app) as c:
        body = {
            "model": "echo",
            "input": {"text": "hello", "extra": {"k": 1}},
            "params": None,
            "stream": False,
        }
        resp = c.post("/inference", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "echo"
        assert data["output"]["text"] == "hello"
        assert data["output"]["extra"] == {"k": 1}


def test_inference_unknown_model():
    with TestClient(app) as c:
        body = {
            "model": "unknown_model",
            "input": {},
        }
        resp = c.post("/inference", json=body)
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "model_not_found"


def test_inference_not_implemented_stubs_return_501():
    with TestClient(app) as c:
        for model_id in [
            "glm4-9b",
            "glm4-z1-9b",
            "llama3.1-nemotron-8b",
            "mistral-small-24b",
            "qwen3-8b",
        ]:
            body = {
                "model": model_id,
                "input": {"markdown": "# Title\nSome content", "chunk_size": 128},
                "stream": False,
            }
            resp = c.post("/inference", json=body)
            assert resp.status_code == 501, f"expected 501 for {model_id}, got {resp.status_code}, {resp.text}"
            data = resp.json()
            assert data["error"]["code"] == "not_implemented"
