import gc
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define constants
MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_INPUT_FILE = "../input_markdown_linkedin.txt"  # Default input file if not specified in arguments
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

    # Format input for Qwen3 model
    messages = [
        {"role": "system",
         "content": "You are a specialized job data parser that extracts requirements from Markdown."},
        {"role": "user", "content": prompt}
    ]

    # Apply chat template for Qwen3
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Tokenize the text
    model_inputs = tokenizer([text], return_tensors="pt")

    # Create explicit attention mask (1s for all tokens)
    input_ids_length = model_inputs["input_ids"].shape[1]
    attention_mask = torch.ones((1, input_ids_length), dtype=torch.long)
    model_inputs["attention_mask"] = attention_mask

    # Move to the correct device
    device = next(model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=2500,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=False
        )

    # Get only the new tokens
    output_ids = generated_ids[0][model_inputs["input_ids"].shape[1]:].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    # Find the start of the JSON object.
    json_start_index = response.find('{')
    if json_start_index == -1:
        print("No JSON object found in the response.")
        return []

    # Extract from the start of the JSON object to the end of the string
    json_str = response[json_start_index:]

    try:
        data = json.loads(json_str)
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

    # Rough estimate: 1 token ≈ 4 characters
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
    except Exception as e:
        print(f"Error reading input file: {e}")
        exit(1)


def main():
    # Get input Markdown from file
    markdown_content = get_markdown_content(DEFAULT_INPUT_FILE)

    # Chunk the markdown content
    chunks = chunk_markdown(markdown_content, chunk_size=CHUNK_SIZE)

    # Clean up memory before loading model
    gc.collect()
    torch.cuda.empty_cache()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Loading tokenizer and model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model_kwargs = {
        "device_map": "auto",  # Changed from balanced to auto for better allocation
        "trust_remote_code": True,
        "torch_dtype": "auto",  # Use auto instead of explicit float16
    }

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

    # Process each chunk
    all_requirements = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_requirements = process_chunk(model, tokenizer, chunk)
        if chunk_requirements:
            # Extract individual items from the dictionary structure
            if isinstance(chunk_requirements, dict):
                # Flatten all the lists in the dictionary
                for category, items in chunk_requirements.items():
                    if isinstance(items, list):
                        all_requirements.extend(items)
                    elif isinstance(items, str):
                        all_requirements.append(items)
            else:
                all_requirements.extend(chunk_requirements)

    # Deduplicate requirements
    unique_requirements = []
    seen = set()

    for req in all_requirements:
        # Handle string requirements only (since we flattened the structure)
        if isinstance(req, str) and req not in seen:
            seen.add(req)
            unique_requirements.append(req)

    # Output final result
    print(f"\n===== Extracted {len(unique_requirements)} Job Requirements =====")

    # Create a properly structured JSON object
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