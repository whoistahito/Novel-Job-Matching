import json
import os
import time
from openai import OpenAI
import outlines
from typing import List
import time
from pydantic import BaseModel, Field



class Requirements(BaseModel):
    """Normalized requirements schema for job extraction outputs."""
    skills: List[str] = Field(
        default_factory=list,
        description="List of required skills"
    )
    experiences: List[str] = Field(
        default_factory=list,
        description="List of experience requirements"
    )
    qualifications: List[str] = Field(
        default_factory=list,
        description="List of qualifications/certifications"
    )

API_BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY =os.environ["EXTERNAL_LLM_API_KEY"]
MODEL_NAME = "deepseek-ai/deepseek-v3.1-terminus"
INPUT_DIR = "../requirements_extraction/datasets/markdown_dataset"
OUTPUT_DIR = f"../{MODEL_NAME.replace('/', '_')}_results"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not API_KEY:
    raise RuntimeError("NVIDIA_API_KEY environment variable is not set.")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
model = outlines.from_openai(
    client,
    MODEL_NAME
)
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".md"):
        continue

    input_filepath = os.path.join(INPUT_DIR, filename)
    with open(input_filepath, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    prompt = f"""
    You are an expert job requirements extractor. Analyze the following job description and extract ONLY the mandatory requirements.

    Instructions:
    1. Focus ONLY on clearly stated MUST-HAVE requirements (required, essential, necessary)
    2. IGNORE all "nice-to-have", "preferred", or "bonus" skills/qualifications
    3. Be specific and concise - extract exact requirements, not general topics
    4. Categorize each requirement as either:
       - "skills": Technical abilities or tools proficiency (e.g., "Python programming", "project management")
       - "experiences": Work history requirements (e.g., "5+ years in software development", "experience with agile methodologies")
       - "qualifications": Formal education or certifications (e.g., "Bachelor's in CS", "PMP certification")

    FORMAT: Return ONLY JSON with three lists (skills, experiences, qualifications).
    IMPORTANT: Return valid JSON ONLY. No extra text, explanations, or formatting.

    JOB DESCRIPTION:
    {markdown_content}
    """

    print(f"Processing {filename} with model {MODEL_NAME}...")
    start_time = time.time()
    try:
        completion= model(
            prompt,
            Requirements,
            temperature=0.5,
        )
        completion = completion.replace("\n", "")
        completion = completion.replace("<|return|>", "")
        completion = completion.replace("```json", "")
        completion = completion.replace("```", "")
        status_ok = True
        completion = Requirements.model_validate_json(completion).model_dump()
    except Exception as e:
        completion = Requirements().model_dump()
        status_ok = False
    end_time = time.time()
    processing_time_in_seconds = end_time - start_time
    print("Status:", "ok" if status_ok else "error")
    result = {
        "modelId": MODEL_NAME,
        "inputText": markdown_content,
        "result": {"requirements": completion} ,
        "processingTimeInSeconds": processing_time_in_seconds,
    }

    output_filename = os.path.splitext(filename)[0] + ".json"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"Result saved to {output_filepath}")
    time.sleep(1)

