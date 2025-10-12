from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import os
import re
from typing import List, Dict, Set

import outlines
from outlines import Template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema import Requirements

# Define constants
MODEL_ID = "THUDM/GLM-Z1-9B-0414"
INPUT_FILE = "../datasets/input_markdown_linkedin.txt"
CHUNK_SIZE = 12000
OUTPUT_FILE = 'job_requirements.json'


def process_chunk(model, chunk) -> Requirements:
    """Process a single chunk using structured generation into Requirements schema."""
    template = Template.from_file("prompt_template.txt")
    prompt: str = template(chunk=chunk)
    try:
        response = model(prompt, output_type=Requirements, max_new_tokens=200)
        # In glm4-9b.py they attempt to re-validate JSON; we mirror that pattern for consistency
        try:
            return Requirements.model_validate_json(response)
        except Exception:
            # If response already is a Requirements instance or dict
            if isinstance(response, Requirements):
                return response
            if isinstance(response, dict):
                return Requirements(**response)
            return Requirements()
    except Exception as e:
        print(f"Error during generation: {e}")
        return Requirements()


def chunk_markdown(markdown_text, chunk_size=3000) -> List[str]:
    """
    Split markdown text into chunks for processing.

    Args:
        markdown_text: Markdown formatted text
        chunk_size: Maximum token size for each chunk (approximate)

    Returns:
        List of markdown chunks ready for processing
    """
    print("Breaking down the Markdown into manageable chunks...")

    # Rough estimate: 1 token ≈ 4 characters for English text
    chars_per_chunk = chunk_size * 4

    # Try to split on major Markdown elements (headers)
    chunks = re.split(r'(#{1,6}\s+.*?\n)', markdown_text)

    # Recombine to stay under token limit
    result_chunks: List[str] = []
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


def get_markdown_content(input_file) -> str:
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
    print(f"Current Date and Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    # Get input Markdown from file
    print(f"Reading input from: {INPUT_FILE}")
    markdown_content = get_markdown_content(INPUT_FILE)

    # Chunk the markdown content
    chunks = chunk_markdown(markdown_content, chunk_size=CHUNK_SIZE)

    # Clean up memory before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load tokenizer and model
    print(f"Loading tokenizer and model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = outlines.from_transformers(hf_model, tokenizer)

    # Process each chunk
    all_requirements: List[Requirements] = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        req_obj = process_chunk(model, chunk)
        all_requirements.append(req_obj)

    merged: Dict[str, Set[str]] = defaultdict(set)
    for req in all_requirements:
        merged["skills"].update(req.skills)
        merged["experiences"].update(req.experiences)
        merged["qualifications"].update(req.qualifications)

    unique_requirements: Dict[str, List[str]] = {k: sorted(v) for k, v in merged.items()}

    print(f"\n===== Extracted {sum(len(v) for v in unique_requirements.values())} Unique Requirement Items =====")

    result_obj = {"requirements": unique_requirements}
    result_json = json.dumps(result_obj, indent=2)
    print(result_json)

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(result_json)
    print(f"Results saved to {OUTPUT_FILE}")

    # Clean up
    del hf_model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()