import gc
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define constants
MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
DEFAULT_INPUT_FILE = "../input_markdown_linkedin.txt"  # Default input file if not specified in arguments
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
  "skills": ["Python programming", "AWS cloud services", "Docker containerization"],
  "experience": ["3+ years software development", "2+ years with microservices"],
  "qualifications": ["Bachelor's degree in Engineering","AWS Solutions Architect certification"],
}}

IMPORTANT: Return ONLY the JSON object above, no explanations or additional text."""

    messages = [
        {"role": "system",
         "content": "detailed thinking off"},
        {"role": "user", "content": prompt}
    ]

    try:
        # Apply Mistral chat template
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

        # Move to the correct device
        device = next(model.parameters()).device
        model_inputs = model_inputs.to(device)

        # Generate response
        with torch.inference_mode():
            generated_ids = model.generate(
                model_inputs,
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.95,
                do_sample=True
            )

        output_ids = generated_ids[0][model_inputs.shape[1]:].tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        # Extract assistant response
        if isinstance(response, list):
            assistant_response = next((msg['content'] for msg in response if msg['role'] == 'assistant'), None)
            if assistant_response:
                response = assistant_response
            else:
                response = str(response)

        print(f"Raw response: {response[:200]}...")  # Truncated debug print

        response = response.strip()

        try:
            data = json.loads(response)
            print(f"Parsed JSON: {json.dumps(data, indent=2)}")

            # Extract all requirements into a flat list
            all_reqs = []
            for category in ['skills', 'experience', 'qualifications']:
                if category in data and isinstance(data[category], list):
                    all_reqs.extend(data[category])

            return all_reqs

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Attempted to parse: {response}")
            return []

    except Exception as e:
        print(f"Error during generation: {e}")
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **model_kwargs
    )


    # Process each chunk
    all_requirements = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_requirements = process_chunk(model, tokenizer, chunk)
        if chunk_requirements:
            all_requirements.extend(chunk_requirements)

    # Deduplicate requirements
    unique_requirements = []
    seen = set()

    for req in all_requirements:
        # Handle different requirement formats
        if isinstance(req, dict):
            # Convert dict to frozen set of items for hashing
            req_tuple = frozenset((k, str(v)) for k, v in req.items())
            if req_tuple not in seen:
                seen.add(req_tuple)
                unique_requirements.append(req)
        elif isinstance(req, str) and req not in seen:
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