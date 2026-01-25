import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 15


def _extract_score(data: dict) -> float :
    try:
        score = data["result"]["similarityScore"]["score"]
        return float(score)
    except Exception:
        return 0


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "Linkedin_comparison" / "results_qwen3-8b"
    output_dir = Path(__file__).parent

    scores: list[float] = []
    missing: list[str] = []

    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        score = _extract_score(data)
        if score is None:
            missing.append(json_file.name)
            continue

        scores.append(score)

    if not scores:
        raise RuntimeError(f"No similarityScore found in {data_dir}")

    scores_arr = np.asarray(scores, dtype=float)

    if missing:
        print(f"Missing/invalid similarityScore in {len(missing)} files: {', '.join(missing)}")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Bucketed Scores
    bucket_edges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 1.0], dtype=float)
    bucket_labels = ["0.0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "≥0.4"]
    
    counts, _ = np.histogram(scores_arr, bins=bucket_edges)
    perc = counts / len(scores_arr) * 100.0

    fig, ax = plt.subplots(figsize=(10, 6))
    
    palette = sns.color_palette("Blues", n_colors=len(counts))
    bars = ax.bar(bucket_labels, counts, color=palette, edgecolor="white", linewidth=1)

    ax.set_xlabel("Similarity Score Range", fontweight="bold")
    ax.set_ylabel("Number of Job Postings", fontweight="bold")
    ax.set_title("Job Matching Performance using qwen3-8b", fontweight="bold", pad=15)
    
    for bar, count, p in zip(bars, counts, perc):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2., 
                height + 0.1,
                f"{int(count)}\n({p:.1f}%)",
                ha="center", 
                va="bottom", 
                fontsize=10, 
                fontweight="bold",
                color="#333333"
            )

    ax.grid(visible=False, axis='x')
    sns.despine(left=True)

    plt.tight_layout()
    bucket_path = output_dir / "bucketed_similarity_scores_qwen3-8b.png"
    plt.savefig(bucket_path, dpi=300, bbox_inches="tight")
    print(f"plot saved: {bucket_path}")
    plt.close()


if __name__ == "__main__":
    main()
