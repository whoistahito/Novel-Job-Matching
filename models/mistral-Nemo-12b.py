import gc
import json
import os
import re

from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define constants
MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"  # Updated to use Mistral-Small-24B
INPUT_FILE = "../input_markdown_linkedin.txt"  # Default input file if not specified in arguments
CHUNK_SIZE = 12000
OUTPUT_FILE = 'job_requirements.json'

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
        {"role": "system",
         "content": "You are a specialized job data parser that extracts requirements from Markdown."},
        {"role": "user", "content": prompt}
    ]

    response = ""
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
                temperature=0.3,
                top_p=0.95,
                top_k=20,
                do_sample=True
            )

        # Get only the new tokens
        output_ids = generated_ids[0][model_inputs.shape[1]:].tolist()
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    except Exception as e:
        print(f"Error during model inference: {e}")
        return []

    # Clean and extract JSON from the response
    try:
        parsed_json = json.loads(response.strip())
        print("Successfully parsed JSON:")
        print(json.dumps(parsed_json, indent=2))

        # Return the parsed JSON directly - we'll handle it as a requirement
        return [parsed_json]

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Problematic JSON text: {response}")
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
    # Authenticate with Hugging Face
    if not authenticate_huggingface():
        print("Exiting due to authentication failure.")
        exit(1)

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

    # Fix pad token issues - make pad token different from EOS token
    pad_token_added = False
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        print(f"Added custom pad token: {PAD_TOKEN}")
        pad_token_added = True

    # Configure model loading parameters
    model_kwargs = {
        "device_map": "balanced",
        "max_memory": {0: "10GiB", 1: "10GiB"},
        "dtype": torch.bfloat16,
    }

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

    # Resize model embeddings if pad token was added
    if pad_token_added:
        model.resize_token_embeddings(len(tokenizer))

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