import json
from pathlib import Path

markdown_folder = Path(__file__).parent.parent.parent / "implementation/evaluation_framework/datasets/markdown_dataset"
results_folder = Path(__file__).parent / "job_description_characteristics"

def add_character_counts():
    """Add character count field to existing JSON files."""

    json_files = [f for f in results_folder.glob("*.json") if f.name != "analysis_summary.json"]
    total_files = len(json_files)

    print(f"Found {total_files} JSON files to update")

    updated_count = 0
    error_count = 0

    for idx, json_file in enumerate(json_files, 1):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            markdown_filename = data.get("filename")
            if not markdown_filename:
                print(f"  Warning: No filename found in {json_file.name}")
                error_count += 1
                continue

            markdown_file = markdown_folder / markdown_filename

            if not markdown_file.exists():
                print(f"  Warning: Markdown file not found: {markdown_filename}")
                error_count += 1
                continue

            # Count characters
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
                character_count = len(markdown_text)

            data["characterCount"] = character_count

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            updated_count += 1


        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")
            error_count += 1
            continue

    print(f"Total : {total_files}")
    print(f"Errors: {error_count}")

    # Calculate statistics
    if updated_count > 0:
        character_counts = []

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "characterCount" in data:
                        character_counts.append(data["characterCount"])
            except Exception:
                continue


if __name__ == "__main__":
    add_character_counts()

