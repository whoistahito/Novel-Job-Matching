from collections import defaultdict
import gc
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "THUDM/GLM-4-9B-0414"
INPUT_FILE = "../input_markdown_linkedin.txt"  # Default input file if not specified in arguments
CHUNK_SIZE = 12000
OUTPUT_FILE = 'job_requirements.json'


def process_chunk(model, tokenizer, chunk):
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

    # Format input for GLM-4 model
    message = [
        {"role": "system",
         "content": "You are a helpful assistant. Your Task is to extract job requirements from provided text."},
        {"role": "user", "content": prompt}
    ]

    # Handle GLM-4 tokenizer format specifically
    try:

        inputs = tokenizer.apply_chat_template(
            message,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)

        # Move to the correct device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": 8000,
            "do_sample": False,
        }
        # Generate response
        with torch.inference_mode():
            output = model.generate(**generate_kwargs)

        # Decode the response
        full_output = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Extract the model's response
        response = full_output.split("<|assistant|>\n")[-1].strip()

    except Exception as e:
        print(f"Error during generation: {e}")
        # Fallback to simpler approach
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=10000,
                temperature=0.1
            )
        response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Response was: {response}")
        return []


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

    # Clean up memory before loading model
    gc.collect()
    torch.cuda.empty_cache()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load tokenizer and model
    print(f"Loading tokenizer and model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16
    )

    # Process each chunk
    all_requirements = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_requirements = process_chunk(model, tokenizer, chunk)
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
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
