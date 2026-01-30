import torch


def test_compute_maxsim_empty_lists_returns_zero():
    from inference.similarity_search import compute_maxsim

    class FakeModel:
        def encode(self, items, convert_to_tensor=True):
            return torch.zeros((len(items), 3))

    m = FakeModel()
    assert compute_maxsim([], ["a"], m) == 0.0
    assert compute_maxsim(["a"], [], m) == 0.0


def test_compute_maxsim_uses_rowwise_max_then_mean(monkeypatch):
    from inference import similarity_search

    class FakeModel:
        def encode(self, items, convert_to_tensor=True):
            return torch.zeros((len(items), 3))

    def fake_cos_sim(job, user):
        # shape (job, user)
        return torch.tensor(
            [
                [0.1, 0.7, 0.2],
                [0.3, 0.4, 0.9],
            ]
        )

    monkeypatch.setattr(similarity_search.util, "cos_sim", fake_cos_sim)

    score = similarity_search.compute_maxsim(["u1", "u2", "u3"], ["j1", "j2"], FakeModel())
    assert abs(score - 0.8) < 1e-6


def test_compute_similarity_default_weights(monkeypatch):
    from inference import similarity_search
    from inference.api_schema import Requirements, UserProfile

    monkeypatch.setattr(similarity_search, "get_model", lambda: object())

    def fake_compute_maxsim(user_items, job_items, model):
        mapping = {"skills": 0.6, "experiences": 0.2, "qualifications": 0.0}
        if user_items is user_profile.skills:
            return mapping["skills"]
        if user_items is user_profile.experiences:
            return mapping["experiences"]
        return mapping["qualifications"]

    monkeypatch.setattr(similarity_search, "compute_maxsim", fake_compute_maxsim)

    user_profile = UserProfile(skills=["python"], experiences=["2y"], qualifications=["bs"])
    req = Requirements(skills=["python"], experiences=["2y"], qualifications=[])

    result = similarity_search.compute_similarity(user_profile, req)
    assert abs(result.score - (0.6 * 0.5 + 0.2 * 0.3 + 0.0 * 0.2)) < 1e-9


def test_compute_similarity_custom_weights_missing_keys(monkeypatch):
    from inference import similarity_search
    from inference.api_schema import Requirements, UserProfile

    monkeypatch.setattr(similarity_search, "get_model", lambda: object())
    monkeypatch.setattr(similarity_search, "compute_maxsim", lambda *args, **kwargs: 1.0)

    user_profile = UserProfile(skills=["a"], experiences=["b"], qualifications=["c"])
    req = Requirements(skills=["a"], experiences=["b"], qualifications=["c"])

    result = similarity_search.compute_similarity(user_profile, req, weights={"skills": 1.0})
    assert result.score == 1.0
