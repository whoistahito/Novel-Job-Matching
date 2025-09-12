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


def _extract_json(text: str):
    """Try to extract a JSON object from arbitrary text."""
    text = text.strip()
    # Fast path: direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass
    # Heuristic: find first '{' and last '}' and try to parse that slice
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def process_chunk(model, tokenizer, chunk):
    """Process a single chunk of Markdown with the LLM and return a structured dict."""
    prompt = (
        "You are an expert job requirements extractor. Analyze the following text and extract ONLY specific, actionable job requirements.\n\n"
        "RULES:\n"
        "- Extract concrete requirements only (skills, experience years, certifications, education)\n"
        "- Skip: company overview, benefits, culture, responsibilities, \"nice-to-have\" items\n"
        "- Be precise with experience requirements (e.g., \"3+ years Python\" not just \"Python experience\")\n"
        "- Include specific technologies, tools, and methodologies mentioned\n"
        "- Only extract what is explicitly required, not preferred\n\n"
        f"TEXT TO ANALYZE:\n{chunk}\n\n"
        "OUTPUT FORMAT (JSON only, no other text):\n"
        "{\n"
        "  \"skills\": [\"Python programming\", \"AWS cloud services\", \"Docker\"],\n"
        "  \"experience\": [\"3+ years software development\", \"2+ years with microservices\"],\n"
        "  \"qualifications\": [\"Bachelor's degree in Engineering\", \"AWS Solutions Architect certification\"]\n"
        "}\n\n"
        "IMPORTANT: Return ONLY the JSON object above, no explanations or additional text."
    )

    messages = [
        {"role": "system", "content": "detailed thinking off"},
        {"role": "user", "content": prompt},
    ]

    try:
        # Apply chat template with generation prompt
        apply_chat = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat):
            try:
                model_inputs = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                )
            except TypeError:
                model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        else:
            # Fallback naive chat formatting
            chat_text = "".join(
                f"<|{m['role']}|>\n{m['content']}<|endoftext|>\n" for m in messages
            ) + "<|assistant|>\n"
            model_inputs = tokenizer(chat_text, return_tensors="pt").input_ids

        # Move to the correct device
        device = next(model.parameters()).device
        model_inputs = model_inputs.to(device)

        # Generate response (Reasoning OFF -> greedy decoding)
        with torch.inference_mode():
            generated_ids = model.generate(
                model_inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        output_ids = generated_ids[0][model_inputs.shape[1]:]
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Strip any thinking tags if present
        if "</think>" in response:
            think_end_index = response.find("</think>")
            response = response[think_end_index + len("</think>"):].strip()

        # Parse JSON
        data = _extract_json(response)
        if not isinstance(data, dict):
            return None

        # Accept either top-level fields or nested under "requirements"
        if "requirements" in data and isinstance(data["requirements"], dict):
            data = data["requirements"]

        out = {
            "skills": data.get("skills", []) if isinstance(data.get("skills"), list) else [],
            "experience": data.get("experience", []) if isinstance(data.get("experience"), list) else [],
            "qualifications": data.get("qualifications", []) if isinstance(data.get("qualifications"), list) else [],
        }
        return out

    except Exception as e:
        print(f"Error during generation: {e}")
        return None


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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load tokenizer and model
    print(f"Loading tokenizer and model from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Ensure a pad token id is set
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
        # "trust_remote_code": True,  # enable if needed
    }

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **model_kwargs
    )

    # Process each chunk
    all_requirements = {"skills": [], "experience": [], "qualifications": []}

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_requirements = process_chunk(model, tokenizer, chunk)
        if isinstance(chunk_requirements, dict):
            for category in ["skills", "experience", "qualifications"]:
                if isinstance(chunk_requirements.get(category), list):
                    all_requirements[category].extend(chunk_requirements[category])

    # Deduplicate requirements within each category
    for category in all_requirements:
        seen = set()
        unique_items = []
        for item in all_requirements[category]:
            if isinstance(item, str):
                key = item
            elif isinstance(item, dict):
                key = json.dumps(item, sort_keys=True)
            else:
                key = str(item)
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        all_requirements[category] = unique_items

    total_count = sum(len(items) for items in all_requirements.values())
    print(f"\n===== Extracted {total_count} Job Requirements =====")

    # Create a properly structured JSON object
    result_obj = {
        "skills": all_requirements["skills"],
        "experience": all_requirements["experience"],
        "qualifications": all_requirements["qualifications"],
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()