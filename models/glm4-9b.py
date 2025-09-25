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
INPUT_FILE = "../input_markdown_linkedin.txt"
CHUNK_SIZE = 12000
OUTPUT_FILE = 'job_requirements.json'


def process_chunk(model, chunk) -> Requirements:
    """Process a single chunk of Markdown with the LLM."""
    prompt = f"""You are an expert job requirements extractor. Analyze the following job description and extract ONLY the mandatory requirements. 

    Instructions:
    1. Focus ONLY on clearly stated MUST-HAVE requirements (required, essential, necessary)
    2. IGNORE all "nice-to-have", "preferred", or "bonus" skills/qualifications
    3. Be specific and concise - extract exact requirements, not general topics
    4. Categorize each requirement as either:
       - "skills": Technical abilities or tools proficiency (e.g., "Python programming", "project management")
       - "experiences": Work history requirements (e.g., "5+ years in software development", "experience with agile methodologies") 
       - "qualifications": Formal education or certifications (e.g., "Bachelor's in CS", "PMP certification")

    FORMAT: Return ONLY JSON with three lists (skills, experiences, qualifications).

    JOB DESCRIPTION:
    {chunk}
    """
    try:
        response = model(prompt, output_type=Requirements, max_new_tokens=200)
        print(response)
        return Requirements.model_validate_json(response)
    except Exception as e:
        print(f"Error during generation: {e}")
        return Requirements()


def chunk_markdown(markdown_text, chunk_size=3000) -> list[str]:
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
    model_kwargs = {
        "device_map": "balanced",
        "max_memory": {0: "10GiB", 1: "10GiB"},
        "dtype": torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **model_kwargs
    )
    model = outlines.from_transformers(hf_model, tokenizer)

    all_requirements: List[Requirements] = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_requirements = process_chunk(model, chunk)
        all_requirements.append(chunk_requirements)

    merged_requirements: Dict[str, Set[str]] = defaultdict(set)
    for req in all_requirements:
        merged_requirements["skills"].update(req.skills)
        merged_requirements["experiences"].update(req.experiences)
        merged_requirements["qualifications"].update(req.qualifications)

    # Convert sets back to lists for JSON serialization
    unique_requirements: Dict[str, List[str]] = {
        key: sorted(list(values)) for key, values in merged_requirements.items()
    }

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
