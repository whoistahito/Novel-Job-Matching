from collections import defaultdict
import gc
import json
import os
import re

import outlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema import Requirements

MODEL_ID = "THUDM/GLM-4-9B-0414"
INPUT_FILE = "../input_markdown_linkedin.txt"  # Default input file if not specified in arguments
CHUNK_SIZE = 12000
OUTPUT_FILE = 'job_requirements.json'


def process_chunk(model, chunk):
    """Process a single chunk of Markdown with the LLM."""
    prompt = f"""You are an expert job requirements extractor. Analyze the following text and extract ONLY specific, actionable job requirements.

    RULES:
    - Extract concrete requirements only (skills, experience years, certifications, education)
    - Skip: company overview, benefits, culture, responsibilities, "nice-to-have" items
    - Be precise with experience requirements (e.g., "3+ years Python" not just "Python experience")
    - Include specific technologies, tools, and methodologies mentioned
    - Only extract what is explicitly required, not preferred

    TEXT TO ANALYZE:
    {chunk}

    OUTPUT FORMAT (JSON only, no other text):
    {{
      "skills": ["Python programming", "AWS cloud services", "Docker","Git", "Kubernetes"],
      "experience": ["3+ years software development", "2+ years with microservices"],
      "qualifications": ["Bachelor's degree in Engineering","AWS Solutions Architect certification"],
    }}

    IMPORTANT: Return ONLY the JSON object above, no explanations or additional text."""

    # Handle GLM-4 tokenizer format specifically
    try:
        # Directly call the outlines model
        response = model(prompt, output_type=Requirements, max_new_tokens=1000)
        return response.model_dump() if hasattr(response, "model_dump") else response
    except Exception as e:
        print(f"Error during generation: {e}")
        return {"skills": [], "experience": [], "qualifications": []}

def chunk_markdown(markdown_text, chunk_size=3000):
    """
    Split markdown text into chunks for processing.

    Args:
        markdown_text: Markdown formatted text
        chunk_size: Maximum token size for each chunk (approximate)

    Returns:
        List of markdown chunks ready for processing
    """
    print("Breaking down the Markdown into manageable chunks...")

    chars_per_chunk = chunk_size * 4

    # Try to split on major Markdown elements (headers)
    chunks = re.split(r'(#{1,6}\s+.*?\n)', markdown_text)

    # Recombine to stay under token limit
    result_chunks = []
    current_chunk = ""

    for chunk in chunks:
        if len(current_chunk) + len(chunk) < chars_per_chunk:
            current_chunk += chunk
        else:
            if current_chunk:
                result_chunks.append(current_chunk)
            current_chunk = chunk

    if current_chunk:
        result_chunks.append(current_chunk)

    # If we have no chunks or all chunks are too large, do character-based chunking
    if not result_chunks or min(len(c) for c in result_chunks) > chars_per_chunk:
        result_chunks = [markdown_text[i:i + chars_per_chunk] for i in range(0, len(markdown_text), chars_per_chunk)]

    print(f"Created {len(result_chunks)} chunks for processing")
    return result_chunks


def get_markdown_content(input_file):
    """Get Markdown content from file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        exit(1)


def main():
    # Current date and user info

    # Get input Markdown from file
    print(f"Reading input from: {INPUT_FILE}")
    markdown_content = get_markdown_content(INPUT_FILE)

    # Chunk the markdown content
    chunks = chunk_markdown(markdown_content, chunk_size=CHUNK_SIZE)

    # Clean up memory before loading hf_model
    gc.collect()
    torch.cuda.empty_cache()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load tokenizer and hf_model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16
    )
    model = outlines.from_transformers(hf_model, tokenizer)

    # Process each chunk
    all_requirements = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_requirements = process_chunk(model, chunk)
        if chunk_requirements:
            all_requirements.extend(chunk_requirements)

    merged_requirements = defaultdict(set)

    # Merge all requirements (only dictionaries)
    for req in all_requirements:
        for key, values in req.items():
            merged_requirements[key].update(values)  # Add elements to the set (no duplicates)

    unique_requirements = {key: list(values) for key, values in merged_requirements.items()}

    print(f"\n===== Extracted {len(unique_requirements)} Job Requirements =====")

    result_obj = {
        "requirements": unique_requirements,
    }

    result_json = json.dumps(result_obj, indent=2)
    print(result_json)

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(result_json)
    print(f"Results saved to {OUTPUT_FILE}")

    # Clean up
    del hf_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
