import json
from pathlib import Path
import time

from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=""
)

results_folder = Path(__file__).parent / "job_description_characteristics"
valid_industries = ["Tech", "Legal & Public Sector", "Healthcare", "Engineering",
                    "Education", "Business & Finance", "Logistics"]

INDUSTRY_PROMPT = """You are an expert at analyzing job descriptions and classifying them into industries.

in the text provided to you,there is a job description, read it and determine which ONE of the following industries it belongs to:
- Tech
- Legal & Public Sector
- Healthcare
- Engineering
- Education
- Business & Finance
- Logistics

Respond with ONLY with one word and it should be one of the options above.

If the job description doesn't clearly fit into any of these categories, use "Other".
Do not include any other text, explanation, or markdown formatting."""


def classify_industry(text: str) -> str:
    """Classify the industry of a job description using the LLM API."""
    response_text = "NONE"
    try:

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": INDUSTRY_PROMPT},
                {"role": "user", "content": f"Classify the industry for this job description:\n\n{text}"}
            ],
            temperature=0.3,
            top_p=1,
            max_tokens=4120,
            stream=False
        )

        response_text = completion.choices[0].message.content.strip()

        if response_text in valid_industries:
            return response_text
        else:
            return "Other"
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        print(response_text)
        return "Other"


def process_job_descriptions():
    """Add industry classification field to existing JSON files."""

    # Get all JSON files (excluding summary)
    json_files = [f for f in results_folder.glob("*.json") if f.name != "analysis_summary.json"]
    total_files = len(json_files)

    if total_files == 0:
        print("No JSON files found in the job_description_characteristics folder.")
        return

    print(f"Found {total_files} JSON files to update\n")

    updated_count = 0
    error_count = 0

    for idx, json_file in enumerate(json_files, 1):
        print(f"Processing {idx}/{total_files}: {json_file.stem}")

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            markdown_text = data.get("input_text", "")

            if not markdown_text:
                print(f"  Warning: No input text found in {json_file.name}")
                error_count += 1
                continue

            industry = classify_industry(markdown_text)
            time.sleep(1.5)

            data["industry"] = industry

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"  Industry: {industry}")

            updated_count += 1

            if idx % 10 == 0:
                print(f"  Progress: {idx}/{total_files} files processed")

        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")
            error_count += 1
            continue

    print(f"Total: {total_files}")
    print(f"Successful: {updated_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    process_job_descriptions()
