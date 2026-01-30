"""
This code aims to show how maxsim could be integrated in job matching system.
This is an exploration of how the algorithm works.
"""
import json
from pathlib import Path
from typing import Dict, List
import torch
from sentence_transformers import SentenceTransformer, util

JOB_FILE = "requirements_extraction/llm_responses/qwen3-8b_results/indeed_1af9d03f54a7b00b.json"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_requirements(obj: Dict) -> Dict[str, List[str]]:
    return obj.get("result", {}).get("requirements", {}) or obj.get("profile", {})


def compute_maxsim(user_items: List[str], job_items: List[str], model) -> float:
    """
    Computes the MaxSim score: for each job item, find the best matching user item.
    Returns the average of these best match scores.
    """
    if not user_items or not job_items:
        return 0.0
    
    # Encode lists
    user_embeddings = model.encode(user_items, convert_to_tensor=True)
    job_embeddings = model.encode(job_items, convert_to_tensor=True)
    
    # cosine similarity matrix
    cosine_scores = util.cos_sim(job_embeddings, user_embeddings)
    max_scores, _ = torch.max(cosine_scores, dim=1)

    return max_scores.mean().item()


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    user_profile_path = base_dir / "user_profile.json"
    job_file_path = base_dir / JOB_FILE

    user_data = load_json(user_profile_path)
    user_req = get_requirements(user_data)

    job_data = load_json(job_file_path)
    job_req = get_requirements(job_data)

    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    common_fields = sorted(set(user_req.keys()) & set(job_req.keys()))
    
    overall_scores = []
    
    for field in common_fields:
        user_items = user_req.get(field, [])
        job_items = job_req.get(field, [])

        if not isinstance(user_items, list):
            raise(f"{field} in user profile is not a list ")
        if not isinstance(job_items, list):
            raise(f"{field} in job requirements is not a list ")

        score = compute_maxsim(user_items, job_items, model)
        overall_scores.append(score)
        
        print(f"{field.capitalize()}: {score:.4f}")
        
    if overall_scores:
        avg_score = sum(overall_scores) / len(overall_scores)
        print("-" * 50)
        print(f"Overall Match Score: {avg_score:.4f}")


if __name__ == "__main__":
    main()
