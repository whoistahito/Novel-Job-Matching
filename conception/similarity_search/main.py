import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def to_text(value: List[str]) -> str:
    return ", ".join(value)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_requirements(obj: Dict) -> Dict[str, List[str]]:
    return obj.get("result", {}).get("requirements", {}) or obj.get("profile", {})


def create_embeddings(model: SentenceTransformer, requirements: Dict[str, List[str]]) -> Dict[str, List[float]]:
    return {
        field: model.encode([to_text(values) or " "], normalize_embeddings=True)[0]
        for field, values in requirements.items()
    }


def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    return float(cosine_similarity([embedding1], [embedding2])[0][0])


def compute_job_similarity(
        req_path: Path,
        model: SentenceTransformer,
        user_embeddings: Dict[str, List[float]]
) -> Tuple[float, str, Dict[str, float]]:
    job_data = load_json(req_path)
    job_requirements = get_requirements(job_data)
    job_embeddings = create_embeddings(model, job_requirements)
    common_fields = sorted(set(user_embeddings.keys()) & set(job_embeddings.keys()))

    field_scores = {
        field: calculate_similarity(user_embeddings[field], job_embeddings[field])
        for field in common_fields
    }

    overall_score = sum(field_scores.values()) / len(field_scores) if field_scores else 0.0

    return overall_score, req_path.name, field_scores


def print_results(results: List[Tuple[float, str, Dict[str, float]]]) -> None:
    for overall_score, filename, field_scores in results:
        fields_summary = " ".join(
            f"{field}={score:.3f}"
            for field, score in sorted(field_scores.items())
        )
        print(f"{filename}: overall={overall_score:.3f} {fields_summary}")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    user_profile_path = base_dir / "user_profile.json"
    requirements_dir = base_dir / "requirements_extraction" / "llm_responses" / "qwen3-8b_results"

    if not user_profile_path.exists():
        print(f"Missing user profile: {user_profile_path}", file=sys.stderr)
        sys.exit(1)
    if not requirements_dir.exists():
        print(f"Missing requirements dir: {requirements_dir}", file=sys.stderr)
        sys.exit(1)

    user_data = load_json(user_profile_path)
    user_requirements = get_requirements(user_data)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    user_embeddings = create_embeddings(model, user_requirements)

    job_files = sorted(requirements_dir.glob("*.json"))
    print(job_files[0])
    results = [
        compute_job_similarity(job_file, model, user_embeddings)
        for job_file in job_files
    ]

    results.sort(key=lambda x: x[0], reverse=True)
    print_results(results)


if __name__ == "__main__":
    main()
