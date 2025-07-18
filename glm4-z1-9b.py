import os
import torch
import gc
import json
import re
from datetime import datetime
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define constants
MODEL_ID = "THUDM/GLM-Z1-9B-0414"
DEFAULT_INPUT_FILE = "input_markdown_linkedin.txt"  # Default input file if not specified in arguments


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
                max_new_tokens=1000,
                temperature=0.1,
                do_sample=False
            )

        # Decode the response
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract the model's response
        response = full_output.split("<|assistant|>\n")[-1].strip()

    except Exception as e:
        print(f"Error during generation: {e}")
        # Fallback to simpler approach
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.1
            )
        response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    if "```json" in response:
        # Extract content between ```json and ``` markers
        start_marker = "```json"
        end_marker = "```"
        start_index = response.find(start_marker) + len(start_marker)
        end_index = response.find(end_marker, start_index)

        if end_index != -1:
            json_str = response[start_index:end_index].strip()
        else:
            # Handle case where closing marker might be missing
            json_str = response[start_index:].strip()
    else:
        # Fallback to the original logic
        if response.startswith("json"):
            json_str = response.strip().removeprefix("json").removesuffix("").strip()
        else:
            raise ValueError("Input string does not contain valid JSON markup (no ```json or json prefix found)")

    try:
        data = json.loads(json_str)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        if isinstance(data, dict) and "requirements" in data:
            return data["requirements"]
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
    print(f"Current Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract job requirements from Markdown using GLM-4')
    parser.add_argument('--chunk_size', type=int, default=12000, help='Maximum token size per chunk')
    parser.add_argument('--output', type=str, default='job_requirements.json', help='Output JSON file')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE,
                        help=f'Input Markdown file (default: {DEFAULT_INPUT_FILE})')
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

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # Load tokenizer and model
    print(f"Loading tokenizer and model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="balanced",
        max_memory={0: "9GB", 1: "9GB"},
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    )

    # Check GPU status after model loading
    print_gpu_info("GPU Status After Model Loading")

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
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result_json)
    print(f"Results saved to {args.output}")

    # Clean up
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_info("GPU Status After Cleanup")


if __name__ == "__main__":
    main()