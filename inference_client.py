"""
Batch client for POST /inference.
- Iterates over Markdown files in a directory (default: markdown_dataset) or a single file
- Calls one or more models for each file
- Writes JSONL output with fields: model, filename, input_text, requirements
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Iterable, List, Dict, Any

import requests

# Defaults
DEFAULT_URL = "http://127.0.0.1:8000/inference"
DEFAULT_MODELS = [
    # Put echo first so there's at least one working model without HF deps
    "echo",
    "glm4-9b",
    "glm4-z1-9b",
    "llama3.1-nemotron-8b",
    "mistral-small-24b",
    "qwen3-8b",
]
DEFAULT_INPUT = Path("markdown_dataset")
DEFAULT_OUTPUT = Path("inference_results.jsonl")
DEFAULT_CHUNK_SIZE = 12000
DEFAULT_TIMEOUT = 120.0


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inference over Markdown files to JSONL results")
    p.add_argument("--url", default=DEFAULT_URL, help="Inference endpoint URL")
    p.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model IDs to call (space-separated). If omitted, uses a default list",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to a Markdown file or a directory containing .md files",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to JSONL output file (one JSON object per line)",
    )
    p.add_argument("--grouped-output", type=Path, default=None, help="Optional path to write grouped JSON by file")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    p.add_argument("--limit", type=int, default=0, help="Limit number of files processed (0 = no limit)")
    p.add_argument(
        "--params",
        type=str,
        default=None,
        help="Optional JSON string for the 'params' field (e.g., '{\"temperature\":0.2}' or '{\"mock_text\":\"...\"}')",
    )
    return p.parse_args(argv)


def iter_markdown_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() == ".md":
            yield path
        return
    if not path.is_dir():
        return
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix.lower() == ".md":
            yield p


def build_input_payload(model: str, markdown: str, chunk_size: int) -> dict:
    m = model.lower()
    if m == "echo":
        return {"text": markdown, "extra": {"chunk_size": chunk_size}}
    return {"markdown": markdown, "chunk_size": chunk_size}


def parse_params(params_str: str | None) -> dict | None:
    if not params_str:
        return None
    try:
        return json.loads(params_str)
    except Exception as e:
        print(f"Warning: could not parse --params JSON ({e}); ignoring.", file=sys.stderr)
        return None


def call_inference(
        url: str,
        model: str,
        markdown: str,
        chunk_size: int,
        timeout: float,
        params: dict | None,
) -> tuple[str, dict | None, str | None]:
    payload: Dict[str, Any] = {
        "model": model,
        "input": build_input_payload(model, markdown, chunk_size),
        "stream": False,
    }
    if params is not None:
        payload["params"] = params

    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    except requests.RequestException as e:
        return ("request_error", None, str(e))

    if resp.status_code == 200:
        try:
            data = resp.json()
        except Exception:
            return ("ok", {"raw": resp.text}, None)
        return ("ok", data, None)

    if resp.status_code == 501:
        try:
            err = resp.json()
        except Exception:
            err = {"message": resp.text}
        return ("not_implemented", err, None)

    try:
        err = resp.json()
    except Exception:
        err = {"message": resp.text}
    return (f"http_{resp.status_code}", err, None)


def extract_requirements(response_data: dict | None) -> list:
    if not isinstance(response_data, dict):
        return []
    output = response_data.get("output")
    if isinstance(output, dict) and isinstance(output.get("requirements"), list):
        return output["requirements"]
    # Fallback: try direct array or any common keys
    if isinstance(output, list):
        return output
    if isinstance(response_data.get("requirements"), list):
        return response_data["requirements"]
    return []


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    params = parse_params(args.params)

    files = list(iter_markdown_files(args.input))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"No Markdown files found at: {args.input}", file=sys.stderr)
        return 2

    # Prepare output file (append mode so you can run multiple times)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_fh = args.output.open("a", encoding="utf-8")

    grouped: Dict[str, Dict[str, Any]] = {}

    total = 0
    start_all = time.perf_counter()
    try:
        for i, md_path in enumerate(files, 1):
            markdown = md_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not markdown:
                continue

            if args.grouped_output:
                grouped.setdefault(md_path.name, {"input_text": markdown, "results": {}})

            for model in args.models:
                t0 = time.perf_counter()
                status, data, req_err = call_inference(
                    args.url, model, markdown, args.chunk_size, args.timeout, params
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                requirements = extract_requirements(data) if status == "ok" else []

                record: Dict[str, Any] = {
                    "model": model,
                    "filename": md_path.name,
                    "input_text": markdown,
                    "requirements": requirements,
                    "status": status,
                    "time_ms": round(elapsed_ms, 2),
                }
                if req_err:
                    record["error"] = req_err
                elif status != "ok":
                    record["error"] = data

                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

                if args.grouped_output:
                    grouped[md_path.name]["results"][model] = {
                        "requirements": requirements,
                        "status": status,
                        "time_ms": round(elapsed_ms, 2),
                        **({"error": req_err} if req_err else ({} if status == "ok" else {"error": data})),
                    }

            # Lightweight progress
            if i % 10 == 0:
                print(f"Processed {i}/{len(files)} files...", file=sys.stderr)
    finally:
        out_fh.close()

    # Write grouped output if requested
    if args.grouped_output:
        args.grouped_output.parent.mkdir(parents=True, exist_ok=True)
        with args.grouped_output.open("w", encoding="utf-8") as gf:
            json.dump(grouped, gf, ensure_ascii=False, indent=2)

    elapsed_all = time.perf_counter() - start_all
    print(
        f"Done. Wrote {total} results for {len(files)} files and {len(args.models)} models to {args.output} in {elapsed_all:.1f}s",
        file=sys.stderr,
    )
    if args.grouped_output:
        print(f"Grouped results written to {args.grouped_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
