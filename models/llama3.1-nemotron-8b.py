import argparse
from datetime import datetime
import gc
import json
import os
import re

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define constants
MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
DEFAULT_INPUT_FILE = "../input_markdown_linkedin.txt"  # Default input file if not specified in arguments


def print_gpu_info(label="Current GPU Status"):
    """Print information about available GPUs and their memory usage."""
    print(f"\n===== {label} =====")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        free = total_memory - reserved

        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total memory: {total_memory:.2f} GB")
        print(f"  Reserved memory: {reserved:.2f} GB")
        print(f"  Allocated memory: {allocated:.2f} GB")
        print(f"  Free memory: {free:.2f} GB")
    print("=" * (len(label) + 14))


def process_chunk(pipeline, chunk, enable_thinking=True):
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
        {"role": "system", "content": "You are a precise job requirements extraction tool. Output only valid JSON with the exact format requested."},
        {"role": "user", "content": prompt}
    ]

    try:
        # Generate response
        outputs = pipeline(messages, max_new_tokens=2048, do_sample=True, temperature=0.3, top_p=0.9)
        response = outputs[0]['generated_text']

        # Extract assistant response
        if isinstance(response, list):
            assistant_response = next((msg['content'] for msg in response if msg['role'] == 'assistant'), None)
            if assistant_response:
                response = assistant_response
            else:
                response = str(response)

        print(f"Raw response: {response[:200]}...")  # Truncated debug print

        # Simple JSON extraction
        response = response.strip()

        # Remove any markdown formatting
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()

        # Find JSON object
        json_start = response.find('{')
        json_end = response.rfind('}') + 1

        if json_start == -1 or json_end == 0:
            print("No JSON found in response")
            return []

        json_str = response[json_start:json_end]

        try:
            data = json.loads(json_str)
            print(f"Parsed JSON: {json.dumps(data, indent=2)}")

            # Extract all requirements into a flat list
            all_reqs = []
            for category in ['skills', 'experience', 'qualifications', 'education']:
                if category in data and isinstance(data[category], list):
                    all_reqs.extend(data[category])

            return all_reqs

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Attempted to parse: {json_str}")
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
    # Current date and user info
    print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract job requirements from Markdown using Llama-3.1-Nemotron-Nano-8B-v1')
    parser.add_argument('--chunk_size', type=int, default=12000, help='Maximum token size per chunk')
    parser.add_argument('--output', type=str, default='job_requirements.json', help='Output JSON file')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE,
                        help=f'Input Markdown file (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--enable_thinking', action='store_true', default=True,
                        help='Enable thinking mode for better reasoning (default: True)')
    parser.add_argument('--context_length', type=int, default=32768,
                        help='Maximum context length (default: 32768, max: 131072)')
    args = parser.parse_args()

    # Get input Markdown from file
    print(f"Reading input from: {args.input}")
    markdown_content = get_markdown_content(args.input)

    # Chunk the markdown content
    chunks = chunk_markdown(markdown_content, chunk_size=args.chunk_size)

    # Check available GPUs
    print_gpu_info("Available GPUs")

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
        "torch_dtype": torch.bfloat16,
        "device_map": "auto"
    }

    pipeline_kwargs = {}
    if args.enable_thinking:
        pipeline_kwargs.update({
            "temperature": 0.6,
            "top_p": 0.95,
        })
    else:
        pipeline_kwargs["do_sample"] = False

    pipeline = transformers.pipeline(
       "text-generation",
       model=MODEL_ID,
       tokenizer=tokenizer,
       max_new_tokens=args.context_length,
       **pipeline_kwargs,
       **model_kwargs
    )

    # Check GPU status after model loading
    print_gpu_info("GPU Status After Model Loading")

    # Process each chunk
    all_requirements = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_requirements = process_chunk(pipeline, chunk, args.enable_thinking)
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
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result_json)
    print(f"Results saved to {args.output}")

    # Clean up
    del pipeline, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_info("GPU Status After Cleanup")


if __name__ == "__main__":
    main()