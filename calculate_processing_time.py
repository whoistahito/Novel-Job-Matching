import json
from pathlib import Path
from typing import List, Tuple, Optional

DIR = "llama3-8b-instruct_results"                              # Directory containing JSON files
FIELD_NAME = "processingTimeSeconds"        # Name of field to add
PRECISION = 3                          # Decimal places for seconds
DRY_RUN = False                        # Compute/show changes, don't write files

def load_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Skipping invalid JSON {p}: {e}")
        return None

def write_json(p: Path, data: dict) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")
    tmp.replace(p)

def collect_files(dir_path: Path) -> List[Tuple[Path, float]]:
    files = []
    for p in dir_path.glob("*.json"):
        try:
            ctime = p.stat().st_ctime
            files.append((p, ctime))
        except Exception as e:
            print(f"Skipping {p}: cannot stat file ({e})")
    files.sort(key=lambda t: (t[1], t[0].name.lower()))
    return files

def main() -> None:
    dir_path = Path(DIR).resolve()
    if not dir_path.is_dir():
        print(f"Not a directory: {dir_path}")
        return

    files = collect_files(dir_path)
    if not files:
        print(f"No JSON files found in {dir_path}")
        return

    prev_ctime: Optional[float] = None
    for p, ctime in files:
        delta: Optional[float] = None
        if prev_ctime is not None:
            raw = ctime - prev_ctime
            # Guard against clock skew or non-monotonic times
            delta = round(raw if raw >= 0 else 0.0, PRECISION)

        data = load_json(p)
        if data is None:
            prev_ctime = ctime
            continue

        data[FIELD_NAME] = delta  # time between previous file and this file; first file gets None

        if DRY_RUN:
            print(f"{p.name}: {FIELD_NAME} = {data[FIELD_NAME]}")
        else:
            write_json(p, data)
            print(f"Updated {p.name}: {FIELD_NAME} = {data[FIELD_NAME]}")

        prev_ctime = ctime


if __name__ == "__main__":
    main()