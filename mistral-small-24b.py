import os
import torch
import gc
import json
import re
from datetime import datetime
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

# Define constants
MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"  # Updated to use Mistral-Small-24B
DEFAULT_INPUT_FILE = "input_markdown_linkedin.txt"  # Default input file if not specified in arguments

# Add pad token configuration to avoid warnings
PAD_TOKEN = "<pad>"  # Define a pad token for Mistral model

# Hugging Face authentication function
def authenticate_huggingface():
    """Authenticate with Hugging Face Hub using token."""
    # Check if token is already in environment variable
    token = os.environ.get("HF_TOKEN")

    if not token:
        # If not in environment, prompt user for token
        print("\n===== Hugging Face Authentication Required =====")
        print("This model requires authentication to access.")
        print("Please visit https://huggingface.co/settings/tokens to create a token if you don't have one.")
        token = input("Enter your Hugging Face token: ")

    try:
        # Attempt to log in with the token
        login(token=token, add_to_git_credential=True)
        print("Successfully authenticated with Hugging Face!")
        # Save token to environment variable for future use in this session
        os.environ["HF_TOKEN"] = token
        return True
    except Exception as e:
        print(f"Authentication failed: {e}")
        return False


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

    # Format input for Mistral model
    messages = [
        {"role": "system", "content": "You are a specialized job data parser that extracts requirements from Markdown."},
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
                top_k=20,
                do_sample=True
            )

        # Get only the new tokens
        output_ids = generated_ids[0][model_inputs.shape[1]:].tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        print(f"Response (first 200 chars): {response[:200]}...")

    except Exception as e:
        print(f"Error during generation: {e}")
        # Fallback to simpler approach
        try:
            # Create simpler input format
            prompt_text = f"""<s>[INST] {prompt} [/INST]"""
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            with torch.inference_mode():
                output = model.generate(
                    inputs.input_ids,
                    max_new_tokens=2048,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    do_sample=True
                )

            # Get only the generated tokens, excluding the input
            input_length = inputs.input_ids.shape[1]
            output_ids = output[0][input_length:].tolist()
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        except Exception as inner_e:
            print(f"Fallback approach also failed: {inner_e}")
            return []

    # Clean and extract JSON from the response
    try:
        # Find the first opening brace
        json_start = response.find('{')
        if json_start == -1:
            print("No JSON object found in the response")
            return []

        # Extract potential JSON content
        json_text = response[json_start:]

        # Use a more robust approach to find the proper JSON object
        # Count braces to find the matching closing brace
        brace_count = 0
        end_pos = -1

        for i, char in enumerate(json_text):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i
                    break

        if end_pos != -1:
            json_text = json_text[:end_pos + 1]

        # Remove code block markers if present
        json_text = json_text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]

        # Parse the JSON
        parsed_json = json.loads(json_text.strip())
        print("Successfully parsed JSON:")
        print(json.dumps(parsed_json, indent=2))

        # Return the parsed JSON directly - we'll handle it as a requirement
        return [parsed_json]

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Problematic JSON text: {json_text if 'json_text' in locals() else 'Not available'}")
        return []
    except Exception as e:
        print(f"Error processing response: {e}")
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
    parser = argparse.ArgumentParser(description='Extract job requirements from Markdown using Qwen3-8B-FP8')
    parser.add_argument('--chunk_size', type=int, default=12000, help='Maximum token size per chunk')
    parser.add_argument('--output', type=str, default='job_requirements.json', help='Output JSON file')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE,
                        help=f'Input Markdown file (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--enable_thinking', action='store_true', default=True,
                        help='Enable thinking mode for better reasoning (default: True)')
    parser.add_argument('--context_length', type=int, default=32768,
                        help='Maximum context length (default: 32768, max: 131072 with YaRN)')
    parser.add_argument('--token', type=str, help='Hugging Face token (optional, will prompt if not provided)')
    args = parser.parse_args()

    # Set token in environment if provided via arguments
    if args.token:
        os.environ["HF_TOKEN"] = args.token

    # Authenticate with Hugging Face
    if not authenticate_huggingface():
        print("Exiting due to authentication failure.")
        exit(1)

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

    # Determine if we need to use YaRN scaling for long contexts
    rope_scaling = None
    if args.context_length > 32768:
        if args.context_length > 131072:
            print("Warning: Requested context length exceeds maximum supported (131072). Capping at 131072.")
            args.context_length = 131072

        # Calculate appropriate YaRN factor (based on context length)
        factor = args.context_length / 32768
        rope_scaling = {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": 32768
        }
        print(f"Enabling YaRN scaling with factor {factor} for context length {args.context_length}")

    # Load tokenizer and model
    print(f"Loading tokenizer and model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Fix pad token issues - make pad token different from EOS token
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        print(f"Added custom pad token: {PAD_TOKEN}")

    # Configure model loading parameters
    model_kwargs = {
        "device_map": "balanced",  # Changed to "balanced" to distribute across GPUs
        "max_memory": {0: "10GiB", 1: "10GiB"},  # Explicitly allocate memory on both GPUs
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,  # Use bfloat16 for better memory efficiency
    }

    # Add rope_scaling if needed
    if rope_scaling:
        model_kwargs["rope_scaling"] = rope_scaling

    # Add quantization for better memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model_kwargs["quantization_config"] = quantization_config

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

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