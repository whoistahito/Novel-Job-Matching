from fastapi.testclient import TestClient

from app.api import app


def main():
    with TestClient(app) as c:
        r = c.get("/models")
        print("/models:", r.status_code, r.json())

        body_echo = {"model": "echo", "input": {"text": "hello", "extra": {"k": 1}}, "stream": False}
        r = c.post("/inference", json=body_echo)
        print("/inference echo:", r.status_code, r.json())

        mock = (
            "Here is the result:\n"
            "```json\n{\n"
            "  \"skills\": [\"Python\", \"AWS\"],\n"
            "  \"experience\": [\"3+ years Python\"],\n"
            "  \"qualifications\": [\"Bachelor's in CS\"]\n"
            "}\n```"
        )
        body_glm4 = {
            "model": "glm4-9b",
            "input": {"markdown": "# Title\nSome content", "chunk_size": 128},
            "params": {"mock_text": mock},
            "stream": False,
        }
        r = c.post("/inference", json=body_glm4)
        print("/inference glm4 mock:", r.status_code, r.json())

        body_stub = {
            "model": "mistral-small-24b",
            "input": {"markdown": "# Title\nSome content", "chunk_size": 128},
            "stream": False,
        }
        r = c.post("/inference", json=body_stub)
        print("/inference mistral stub:", r.status_code, r.json())


if __name__ == "__main__":
    main()
