from collections import defaultdict
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

MODEL_ID = "Qwen/Qwen3-8B"
INPUT_FILE = "../../implementation/evaluation_framework/datasets/input_markdown_linkedin.txt"
CHUNK_SIZE = 12000
OUTPUT_FILE = 'job_requirements.json'


def process_chunk(model, chunk) -> Requirements:
    """Process a single chunk of Markdown with the LLM."""
    template = Template.from_file("prompt_template.txt")
    prompt: str = template(chunk=chunk)
    try:
        response = model(prompt, output_type=Requirements, max_new_tokens=200)
        print(response)
        return Requirements.model_validate_json(response)
    except Exception as e:
        print(f"Error during generation: {e}")
        return Requirements()


def chunk_markdown(markdown_text, chunk_size=3000) -> List[str]:
    print("Breaking down the Markdown into manageable chunks...")
    chars_per_chunk = chunk_size * 4
    chunks = re.split(r'(#{1,6}\s+.*?\n)', markdown_text)
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
    if not result_chunks or min(len(c) for c in result_chunks) > chars_per_chunk:
        result_chunks = [markdown_text[i:i + chars_per_chunk] for i in range(0, len(markdown_text), chars_per_chunk)]
    print(f"Created {len(result_chunks)} chunks for processing")
    return result_chunks


def get_markdown_content(input_file) -> str:
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
    print(f"Reading input from: {INPUT_FILE}")
    markdown_content = get_markdown_content(INPUT_FILE)
    chunks = chunk_markdown(markdown_content, chunk_size=CHUNK_SIZE)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Loading tokenizer and model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    structured_model = outlines.from_transformers(model, tokenizer)

    all_requirements: List[Requirements] = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        all_requirements.append(process_chunk(structured_model, chunk))

    merged: Dict[str, Set[str]] = defaultdict(set)
    for req in all_requirements:
        merged["skills"].update(req.skills)
        merged["experiences"].update(req.experiences)
        merged["qualifications"].update(req.qualifications)

    unique_requirements: Dict[str, List[str]] = {k: sorted(v) for k, v in merged.items()}
    print(f"\n===== Extracted {sum(len(v) for v in unique_requirements.values())} Unique Requirement Items =====")

    result_json = json.dumps({"requirements": unique_requirements}, indent=2)
    print(result_json)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(result_json)
    print(f"Results saved to {OUTPUT_FILE}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
