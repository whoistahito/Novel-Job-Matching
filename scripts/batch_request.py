import json
import os
import time

import requests

API_URL = "http://127.0.0.1:8000/extract"
MODEL_ID = "glm4-9b"
INPUT_DIR = "../implementation/evaluation_framework/datasets/markdown_dataset"
OUTPUT_DIR = f"../implementation/evaluation_framework/llm_responses/{MODEL_ID}_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

headers = {
    "Content-Type": "application/json"
}

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".md"):
        input_filepath = os.path.join(INPUT_DIR, filename)
        with open(input_filepath, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        payload = {
            "modelId": MODEL_ID,
            "inputText": markdown_content
        }

        start_time = time.time()
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        end_time = time.time()
        processing_time_in_seconds = end_time - start_time

        print(f"Processing {filename}...")
        print("Status code:", response.status_code)

        if response.status_code == 200:
            result = {
                "modelId": MODEL_ID,
                "inputText": markdown_content,
                "result": response.json(),
                "processingTimeInSeconds": processing_time_in_seconds
            }
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"Result saved to {output_filepath}")
        else:
            print("Error:", response.text)
