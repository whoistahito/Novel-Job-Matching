import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

THRESHOLD = 0.7

plt.style.use("seaborn-v0_8-paper")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 15


@dataclass(frozen=True)
class JobPoint:
    job_name: str
    score: float
    suitable: bool


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {path.name}")
    return data


def _extract_score(d: dict[str, Any]) -> float | None:
    try:
        score = d["result"]["similarityScore"]["score"]
        return float(score)
    except Exception:
        return None


def _extract_suitable(d: dict[str, Any]) -> bool | None:
    val = d.get("suitable")
    return val if isinstance(val, bool) else None


def _job_sort_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    if stem.startswith("job"):
        try:
            return (int(stem[3:]), name)
        except Exception:
            pass
    return (10**9, name)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "Linkedin_comparison" / "results_qwen3-8b"
    prefs_dir = repo_root / "Linkedin_comparison" / "users_preference"
    output_dir = Path(__file__).parent

    points: list[JobPoint] = []
    missing_scores: list[str] = []
    missing_prefs: list[str] = []
    invalid_prefs: list[str] = []

    result_files = sorted(
        results_dir.glob("*.json"), key=lambda p: _job_sort_key(p.name)
    )

    for result_path in result_files:
        score = _extract_score(_load_json(result_path))
        if score is None:
            missing_scores.append(result_path.name)
            continue

        pref_path = prefs_dir / result_path.name
        if not pref_path.exists():
            missing_prefs.append(result_path.name)
            continue

        suitable = _extract_suitable(_load_json(pref_path))
        if suitable is None:
            invalid_prefs.append(result_path.name)
            continue

        points.append(
            JobPoint(job_name=result_path.stem, score=score, suitable=suitable)
        )

    if not points:
        raise RuntimeError(
            "No data points to plot. Ensure results contain result.similarityScore.score "
            "and users_preference contains suitable: true/false."
        )

    if missing_scores:
        print(
            f"Missing/invalid similarityScore in {len(missing_scores)} files: {', '.join(missing_scores)}"
        )
    if missing_prefs:
        print(
            f"Missing users_preference files for {len(missing_prefs)} jobs: {', '.join(missing_prefs)}"
        )
    if invalid_prefs:
        print(
            f"Missing/invalid 'suitable' flag in {len(invalid_prefs)} files: {', '.join(invalid_prefs)}"
        )

    x = np.arange(1, len(points) + 1)
    y = np.array([p.score for p in points], dtype=float)
    labels = [p.job_name for p in points]
    suitable_mask = np.array([p.suitable for p in points], dtype=bool)

    fig, ax = plt.subplots(figsize=(12, 6))

    x_no, y_no = x[~suitable_mask], y[~suitable_mask]
    x_yes, y_yes = x[suitable_mask], y[suitable_mask]

    ax.scatter(
        x_no,
        y_no,
        s=55,
        alpha=0.85,
        color="#d62728",
        edgecolor="white",
        linewidth=0.7,
        label="not suitable",
    )
    ax.scatter(
        x_yes,
        y_yes,
        s=55,
        alpha=0.85,
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.7,
        label="suitable",
    )

    ax.set_xlabel("Job", fontweight="bold")
    ax.set_ylabel("Similarity Score", fontweight="bold")
    ax.set_title("Similarity Score per Job (qwen3-8b)", fontweight="bold", pad=20)

    ax.axhline(
        THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        label=f"threshold = {THRESHOLD}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylim(0, 1)

    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    out_path = output_dir / "scores_scatter_qwen3-8b.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("plot saved")
    plt.close()


if __name__ == "__main__":
    main()
