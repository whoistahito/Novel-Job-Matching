import json
from pathlib import Path
from typing import Dict, List
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

JOB_FILE = "requirements_extraction/llm_responses/qwen3-8b_results/indeed_1af9d03f54a7b00b.json"


def to_text(value: List[str]) -> str:
    return ", ".join(value)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_requirements(obj: Dict) -> Dict[str, List[str]]:
    return obj.get("result", {}).get("requirements", {}) or obj.get("profile", {})


def create_embeddings(model: SentenceTransformer, requirements: Dict[str, List[str]]) -> Dict[str, List[float]]:
    embeddings = {}
    print("requirements",requirements)

    for field, values in requirements.items():
        text = to_text(values)
        if not text:
            text = ""

        encoded = model.encode([text], normalize_embeddings=True)

        embeddings[field] = encoded[0]

    return embeddings


def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    return float(cosine_similarity([embedding1], [embedding2])[0][0])

def calculate_cross_encoder_similarity(user_text: str, job_text: str, cross_model: CrossEncoder) -> float:
    if not user_text.strip() or not job_text.strip():
        return 0.0
    score = cross_model.predict([(user_text, job_text)])
    return float(score[0])

def compare_profiles(
    user_req: Dict[str, List[str]],
    job_req: Dict[str, List[str]],
) -> List[Dict[str, float]]:
    common_fields = sorted(set(user_req.keys()) & set(job_req.keys()))

    encoder_model = SentenceTransformer("all-MiniLM-L6-v2")

    start = time.time()
    user_embeddings = create_embeddings(encoder_model, user_req)
    job_embeddings = create_embeddings(encoder_model, job_req)
    cosine_scores = {
        field: calculate_cosine_similarity(user_embeddings[field], job_embeddings[field])
        for field in common_fields
    }
    end = time.time()
    print(f"Cosine similarity time: {end - start:.4f} seconds")

    cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    start = time.time()
    cross_encoder_scores = {
        field: calculate_cross_encoder_similarity(
            to_text(user_req.get(field, [])),
            to_text(job_req.get(field, [])),
            cross_model
        )
        for field in common_fields
    }
    end = time.time()
    print(f"Cross encoder similarity time: {end - start:.4f} seconds")

    return [cosine_scores, cross_encoder_scores]

def print_detailed_comparison(filename: str, field_scores: Dict[str, float], method_name: str) -> None:
    print(f"\n{method_name} - Comparison with: {filename}")
    print("=" * 60)

    for field, score in sorted(field_scores.items()):
        print(f"{field:20s}: {score:.4f}")

    if field_scores:
        overall = sum(field_scores.values()) / len(field_scores)
        print("-" * 60)
        print(f"{'Overall Score':20s}: {overall:.4f}")
    else:
        print("No common fields found")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    user_profile_path = base_dir / "user_profile.json"
    job_file_path = base_dir / JOB_FILE

    user_data = load_json(user_profile_path)
    user_requirements = get_requirements(user_data)

    job_data = load_json(job_file_path)
    job_requirements = get_requirements(job_data)


    scores = compare_profiles( user_requirements, job_requirements)

    print_detailed_comparison(job_file_path.name, scores[0], "Cosine Similarity")
    print_detailed_comparison(job_file_path.name, scores[1], "Cross-Encoder")


if __name__ == "__main__":
    main()
