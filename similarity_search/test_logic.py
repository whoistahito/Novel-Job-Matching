"""
This code tests the logic of maxsim for better understanding of its implementation.
It tests what happens if a user is underqualified or overqualified.
"""


from sentence_transformers import SentenceTransformer, util
import torch


def compute_maxsim(user_items, job_items, model):
    if not user_items or not job_items:
        return 0.0

    user_embeddings = model.encode(user_items, convert_to_tensor=True)
    job_embeddings = model.encode(job_items, convert_to_tensor=True)

    cosine_scores = util.cos_sim(job_embeddings, user_embeddings)
    # For each row in the matrix (job requirement), find the max score across columns (user skills)
    max_scores, _ = torch.max(cosine_scores, dim=1)

    # Average the best matches to get coverage score
    return max_scores.mean().item()


print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test case: user is Under-qualified (has less skills or education that job requirements need)
# Expectation: score should be low
job1 = ["SQL", "Java", "Python", "Docker", "AWS"]
user1 = ["Python"]
score1 = compute_maxsim(user1, job1, model)
print(f"\nUnder-qualified:")
print(f"Job Requirements: {job1}")
print(f"User Skills:      {user1}")
print(f"Score: {score1:.4f} (Expected: Low)")

# Test case : user is Over-qualified (has more skills or education that job requirements need)
# Expec : score should be high because user covers all the requirements
job2 = ["Python"]
user2 = ["Docker", "Java", "SQL", "Python", "AWS"]
score2 = compute_maxsim(user2, job2, model)
print(f"\nOver-qualified:")
print(f"Job Requirements: {job2}")
print(f"User Skills:      {user2}")
print(f"Score: {score2:.4f} (Expected: High)")
