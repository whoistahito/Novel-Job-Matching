import os
import torch
import gc
import json
import re
from datetime import datetime
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define constants
MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_INPUT_FILE = "input_markdown.txt"  # Default input file if not specified in arguments


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
    prompt = f"""Extract job requirements from this Markdown text and return them in JSON format.
                Focus only on skills, qualifications, experience, and education requirements.
                If this Markdown text doesn't contain job requirements, return an empty JSON array.

                Markdown text:
                {chunk}
                Return ONLY a valid JSON array of requirements with no additional text, it should look like this:
               {{requirements: [
                "qualifications": ["Description of qualification 1"],
                "experineces": ["Description of experinece 1"],
                "skills":["Description of skill 1"],
                ]}}"""

    # Format input for Qwen3 model
    messages = [
        {"role": "system",
         "content": "You are a specialized job data parser that extracts requirements from Markdown."},
        {"role": "user", "content": prompt}
    ]

    try:
        # Apply chat template for Qwen3
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Enable thinking mode for better reasoning
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

        # Generate response with recommended parameters for thinking mode
        with torch.inference_mode():
            generated_ids = model.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=2048,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                do_sample=True  # Must use sampling, not greedy decoding
            )

        # Get only the new tokens
        output_ids = generated_ids[0][model_inputs["input_ids"].shape[1]:].tolist()

        # Parse thinking content and final response
        try:
            # Look for </think> token (ID 151668 according to docs)
            think_end_index = len(output_ids)
            for i in range(len(output_ids)-1, -1, -1):
                if output_ids[i] == 151668:  # </think> token
                    think_end_index = i
                    break

            thinking_content = tokenizer.decode(output_ids[:think_end_index], skip_special_tokens=True).strip("\n")
            response = tokenizer.decode(output_ids[think_end_index:], skip_special_tokens=True).strip("\n")

            # Print thinking content for debugging
            print("Thinking process:")
            print(thinking_content[:200] + "..." if len(thinking_content) > 200 else thinking_content)

        except ValueError:
            # If </think> not found, use entire output
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    except Exception as e:
        print(f"Error during generation: {e}")
        # Fallback to simpler approach
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # Create model inputs properly
        model_inputs = tokenizer([text], return_tensors="pt")

        # Create explicit attention mask (1s for all tokens)
        input_ids_length = model_inputs["input_ids"].shape[1]
        attention_mask = torch.ones((1, input_ids_length), dtype=torch.long)
        model_inputs["attention_mask"] = attention_mask

        # Move to the correct device
        device = next(model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.inference_mode():
            output = model.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=32768,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                do_sample=True
            )

        # Get only the generated tokens, excluding the input
        input_length = model_inputs["input_ids"].shape[1]
        output_ids = output[0][input_length:].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print("Thinking Content:", thinking_content)
        print("Response:", response)
        return response

    # Extract JSON from response
    # The response might start with </think> or other non-JSON text.
    # Find the start of the JSON object.
    json_start_index = response.find('{')
    if json_start_index == -1:
        print("No JSON object found in the response.")
        return []

    # Extract from the start of the JSON object to the end of the string
    json_str = response[json_start_index:]

    # Clean up potential markdown code block fences that might still exist
    if json_str.strip().startswith("```json"):
        json_str = json_str.strip()[7:]
    if json_str.strip().endswith("```"):
        json_str = json_str.strip()[:-3]

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
    parser = argparse.ArgumentParser(description='Extract job requirements from Markdown using Qwen3-8B-FP8')
    parser.add_argument('--chunk_size', type=int, default=12000, help='Maximum token size per chunk')
    parser.add_argument('--output', type=str, default='job_requirements.json', help='Output JSON file')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE,
                        help=f'Input Markdown file (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--enable_thinking', action='store_true', default=True,
                        help='Enable thinking mode for better reasoning (default: True)')
    parser.add_argument('--context_length', type=int, default=32768,
                        help='Maximum context length (default: 32768, max: 131072 with YaRN)')
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

    # Configure model loading parameters
    model_kwargs = {
        "device_map": "auto",  # Changed from balanced to auto for better allocation
        "trust_remote_code": True,
        "torch_dtype": "auto",  # Use auto instead of explicit float16
    }

    # Add rope_scaling if needed
    if rope_scaling:
        model_kwargs["rope_scaling"] = rope_scaling

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