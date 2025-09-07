from datetime import datetime, timezone
import gc
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define constants
MODEL_ID = "THUDM/GLM-Z1-9B-0414"
INPUT_FILE = "../input_markdown_linkedin.txt"
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
    messages = [
        {"role": "system",
         "content": "You are a specialized job data parser that extracts requirements from Markdown."},
        {"role": "user", "content": prompt}
    ]

    response = ""
    # Handle GLM-4 tokenizer format specifically
    try:
        # Format the messages directly for GLM-4
        chat_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                chat_text += f"{content}\n\n"
            elif role == "user":
                chat_text += f"<|user|>\n{content}<|endoftext|>\n"
            elif role == "assistant":
                chat_text += f"<|assistant|>\n{content}<|endoftext|>\n"

        # Add final assistant prompt
        chat_text += "<|assistant|>\n<think>\n"

        # Tokenize the text
        inputs = tokenizer(chat_text, return_tensors="pt")

        # Move to the correct device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=20000,
                temperature=0.6,
                do_sample=False
            )

        # find the where think token is which is [522, 26779, 29]

        # Decode the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(response)
        if "</think>" in response:
            think_end_index = response.find("</think>")
            response = response[think_end_index + len("</think>"):].strip()

    except Exception as e:
        print(f"Error during model inference: {e}")
        return []

    print(response)
    try:
        data = json.loads(response)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        # Handle flat structure format - return the data directly since it contains skills, experience, qualifications
        if isinstance(data, dict) and any(key in data for key in ["skills", "experience", "qualifications"]):
            return data
        else:
            print("Unexpected JSON format, returning empty list.")
            return []
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

    # Rough estimate: 1 token ≈ 4 characters for English text
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
    print(f"Current Date and Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

    # Get input Markdown from file
    print(f"Reading input from: {INPUT_FILE}")
    markdown_content = get_markdown_content(INPUT_FILE)

    # Chunk the markdown content
    chunks = chunk_markdown(markdown_content, chunk_size=CHUNK_SIZE)

    # Clean up memory before loading model
    gc.collect()
    torch.cuda.empty_cache()

    # Set environment variables
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
    all_requirements = {"skills": [], "experience": [], "qualifications": []}

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_requirements = process_chunk(model, tokenizer, chunk)
        if chunk_requirements:
            # Preserve the structured format instead of flattening
            if isinstance(chunk_requirements, dict):
                # Merge each category while maintaining structure
                for category in ["skills", "experience", "qualifications"]:
                    if category in chunk_requirements and isinstance(chunk_requirements[category], list):
                        all_requirements[category].extend(chunk_requirements[category])

    # Deduplicate requirements within each category
    for category in all_requirements:
        seen = set()
        unique_items = []
        for item in all_requirements[category]:
            if isinstance(item, str) and item not in seen:
                seen.add(item)
                unique_items.append(item)
        all_requirements[category] = unique_items

    # Count total requirements
    total_count = sum(len(items) for items in all_requirements.values())
    print(f"\n===== Extracted {total_count} Job Requirements =====")

    # Create the final structured JSON object
    result_obj = {
        "skills": all_requirements["skills"],
        "experience": all_requirements["experience"],
        "qualifications": all_requirements["qualifications"]
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