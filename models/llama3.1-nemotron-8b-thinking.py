import gc
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define constants
MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
DEFAULT_INPUT_FILE = "../input_markdown_linkedin.txt"
INPUT_FILE = "../input_markdown_linkedin.txt"
CHUNK_SIZE = 12000
OUTPUT_FILE = 'job_requirements.json'


def process_chunk(model, tokenizer, chunk):
    """Process a single chunk of Markdown with the LLM using Reasoning ON mode."""
    prompt = f"""You are an expert job requirements extractor. First, think step-by-step about the requirements listed in the text. Then, based on your thoughts, extract ONLY the specific, actionable job requirements into a JSON object.

RULES:
- Extract concrete requirements only: skills, experience years, certifications, education.
- Skip company descriptions, benefits, culture, responsibilities, and "nice-to-have" items.
- Be precise with experience (e.g., "3+ years Python" not just "Python experience").
- Only extract what is explicitly stated as a requirement.

TEXT TO ANALYZE:
{chunk}

Your response MUST be a single JSON object. It must start with `{{` and end with `}}`. Do NOT include any other text, explanations, or code formatting outside of the JSON.

OUTPUT FORMAT:
{{
  "skills": ["skill A", "skill B"],
  "experience": ["experience requirement A", "experience requirement B"],
  "qualifications": ["qualification A", "qualification B"]
}}
"""
    messages = [
        {"role": "system", "content": "detailed thinking on"},
        {"role": "user", "content": prompt}
    ]

    try:
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        device = next(model.parameters()).device
        model_inputs = model_inputs.to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                model_inputs,
                max_new_tokens=10000,  # Increased token limit to allow for thinking
                temperature=0.6,
                top_p=0.95,
                do_sample=True
            )

        # The rest of the decoding process remains the same
        output_ids = generated_ids[0][model_inputs.shape[1]:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        print(f"Raw response from model: {response[:300]}...")

        # The model will output its reasoning first, which we need to strip away.
        think_end_tag = "</think>"
        if think_end_tag in response:
            # Find the end of the thought process and get the text after it
            json_part = response.split(think_end_tag, 1)[1].strip()
        else:
            json_part = response

        data = json.loads(json_part)

        if not data:
            print("Could not extract valid JSON from the final response part.")
            return None

        print(f"Parsed JSON: {json.dumps(data, indent=2)}")

        # Return a structured dictionary
        return {
            "skills": data.get("skills", []) if isinstance(data.get("skills"), list) else [],
            "experience": data.get("experience", []) if isinstance(data.get("experience"), list) else [],
            "qualifications": data.get("qualifications", []) if isinstance(data.get("qualifications"), list) else [],
        }

    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return None

def chunk_markdown(markdown_text, chunk_size=3000):
    print("Breaking down the Markdown into manageable chunks...")
    chars_per_chunk = chunk_size * 4
    chunks = re.split(r'(#{1,6}\s+.*?\n)', markdown_text)
    result_chunks, current_chunk = [], ""
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < chars_per_chunk:
            current_chunk += chunk
        else:
            if current_chunk: result_chunks.append(current_chunk)
            current_chunk = chunk
    if current_chunk: result_chunks.append(current_chunk)
    if not result_chunks or any(len(c) > chars_per_chunk for c in result_chunks):
        result_chunks = [markdown_text[i:i + chars_per_chunk] for i in range(0, len(markdown_text), chars_per_chunk)]
    print(f"Created {len(result_chunks)} chunks for processing")
    return result_chunks

def get_markdown_content(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        exit(1)

def main():
    print(f"Reading input from: {INPUT_FILE}")
    markdown_content = get_markdown_content(INPUT_FILE)
    chunks = chunk_markdown(markdown_content, chunk_size=CHUNK_SIZE)

    gc.collect()
    torch.cuda.empty_cache()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Loading tokenizer and model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    all_requirements = {"skills": [], "experience": [], "qualifications": []}
    for i, chunk in enumerate(chunks):
        print(f"\n--- Processing chunk {i + 1}/{len(chunks)} ---")
        chunk_requirements = process_chunk(model, tokenizer, chunk)
        if chunk_requirements:
            all_requirements["skills"].extend(chunk_requirements.get("skills", []))
            all_requirements["experience"].extend(chunk_requirements.get("experience", []))
            all_requirements["qualifications"].extend(chunk_requirements.get("qualifications", []))

    # Deduplicate requirements within each category
    for category in all_requirements:
        seen = set()
        unique_items = []
        for item in all_requirements[category]:
            if isinstance(item, str) and item not in seen:
                seen.add(item)
                unique_items.append(item)
        all_requirements[category] = unique_items

    total_count = sum(len(items) for items in all_requirements.values())
    print(f"\n===== Extracted {total_count} Unique Job Requirements =====")

    result_json = json.dumps(all_requirements, indent=2)
    print(result_json)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(result_json)
    print(f"Results saved to {OUTPUT_FILE}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()