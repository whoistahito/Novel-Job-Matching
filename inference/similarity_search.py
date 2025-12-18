
from typing import List, Dict

from sentence_transformers import SentenceTransformer, util
import torch

from api_schema import UserProfile, Requirements, SimilarityScore

# Global model instance to avoid reloading on every call
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_maxsim(user_items: List[str],
                   job_items: List[str], model: SentenceTransformer
                   ) -> float:
    if not user_items or not job_items:
        return 0.0

    user_embeddings = model.encode(user_items, convert_to_tensor=True)
    job_embeddings = model.encode(job_items, convert_to_tensor=True)

    cosine_scores = util.cos_sim(job_embeddings, user_embeddings)

    # For each job requirement (row), find the max score across user items (columns)
    max_scores, _ = torch.max(cosine_scores, dim=1)

    return max_scores.mean().item()


def compute_similarity(
        user_profile: UserProfile,
        extracted_requirements: Requirements,
        weights: Dict[str, float] = None
) -> SimilarityScore:
    if weights is None:
        weights = {"skills": 0.5, "experiences": 0.3, "qualifications": 0.2}

    model = get_model()

    requirement_fields = ['skills', 'experiences', 'qualifications']

    overall_scores: Dict[str, float] = {}

    for field in requirement_fields:
        user_items = getattr(user_profile, field, [])
        job_items = getattr(extracted_requirements, field, [])

        # Compute MaxSim score for this field
        score = compute_maxsim(user_items, job_items, model)
        overall_scores[field] = score

    # Calculate weighted average
    weighted_score = sum(
        overall_scores[field] * weights.get(field, 0.0)
        for field in requirement_fields
    )

    return SimilarityScore(score=weighted_score)
