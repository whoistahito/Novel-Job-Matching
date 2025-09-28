import json

import requests

API_URL = "http://127.0.0.1:8000/extract"

# Read the entire input_markdown_linkedin.txt file
with open("../input_markdown_linkedin.txt", "r", encoding="utf-8") as f:
    markdown_content = f.read()

payload = {
    "modelId": "glm4-9b",
    "inputText": markdown_content
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

print("Status code:", response.status_code)
print("Response:", response.text)
