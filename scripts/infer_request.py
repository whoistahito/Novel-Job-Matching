import json

import requests

API_URL = "http://127.0.0.1:8000/inference"

# Read the entire input_markdown_linkedin.txt file
with open("../input_markdown_linkedin.txt", "r", encoding="utf-8") as f:
    markdown_content = f.read()

payload = {
    "model": "glm4-9b",
    "input": {
        "markdown": markdown_content
    },
    "params": {
        "temperature": 0.2
    },
    "stream": False
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

print("Status code:", response.status_code)
print("Response:", response.text)
