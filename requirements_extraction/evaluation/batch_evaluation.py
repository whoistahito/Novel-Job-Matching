import json
import os
from pathlib import Path
import time
from typing import List, Dict, Any

from evaluation_util import run_evaluation, save_evaluation_results

def evaluate_all_models(models: List[str] = None, gemini_api_key: str = None, sample_size: int = None):
    if not models:
        raise ValueError("No models specified for evaluation.")

    output_dir = "response_evaluation"
    all_results = {}

    for model_id in models:
        results_dir = f"../llm_responses/{model_id}_results"

        if not Path(results_dir).exists():
            print(f"\nSkipping {model_id}: Results directory not found")
            continue

        try:
            start_time = time.time()

            eval_results = run_evaluation(model_id, results_dir, gemini_api_key, sample_size)

            if eval_results:
                results_summary = save_evaluation_results(eval_results, model_id,output_dir)

                all_results[model_id] = results_summary

            elapsed_time = time.time() - start_time
            print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")

        except Exception as e:
            print(f"\nError evaluating {model_id}: {e}")
            import traceback
            traceback.print_exc()

    # Generate comparative report
    if all_results:
        generate_comparative_report(all_results,output_dir)

    return all_results


def generate_comparative_report(all_results: Dict[str, Dict[str, Any]],output_dir: str = "response_evaluation"):
    """Generate a comparative report across all evaluated models."""

    output_dir = Path(output_dir)
    output_file = output_dir / "comparative_report.json"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    comparative_report = {
        "evaluated_models": list(all_results.keys()),
        "metrics_comparison": {},
        "rankings": {}
    }

    # Collect all metric names
    all_metrics = set()
    for model_results in all_results.values():
        all_metrics.update(model_results["metrics_summary"].keys())

    # Compare metrics across models
    for metric_name in all_metrics:
        comparative_report["metrics_comparison"][metric_name] = {}

        for model_id, results in all_results.items():
            if metric_name in results["metrics_summary"]:
                metric_data = results["metrics_summary"][metric_name]
                comparative_report["metrics_comparison"][metric_name][model_id] = {
                    "average_score": metric_data["average_score"],
                    "passed_rate": metric_data["passed_count"] / results["total_test_cases"]
                }

    # Calculate rankings for each metric
    for metric_name in all_metrics:
        scores = []
        for model_id, model_data in comparative_report["metrics_comparison"][metric_name].items():
            scores.append((model_id, model_data["average_score"]))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        comparative_report["rankings"][metric_name] = [
            {"rank": i + 1, "model_id": model_id, "score": score}
            for i, (model_id, score) in enumerate(scores)
        ]

    # Calculate overall ranking (average of all metric scores)
    overall_scores = {}
    for model_id in all_results.keys():
        metric_scores = []
        for metric_name in all_metrics:
            if model_id in comparative_report["metrics_comparison"][metric_name]:
                metric_scores.append(
                    comparative_report["metrics_comparison"][metric_name][model_id]["average_score"]
                )

        if metric_scores:
            overall_scores[model_id] = sum(metric_scores) / len(metric_scores)

    overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    comparative_report["rankings"]["overall"] = [
        {"rank": i + 1, "model_id": model_id, "average_score": score}
        for i, (model_id, score) in enumerate(overall_ranking)
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparative_report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("COMPARATIVE REPORT")
    print(f"{'=' * 60}\n")

    print(f"Evaluated Models: {', '.join(all_results.keys())}\n")

    print("Overall Rankings:")
    print("-" * 60)
    for rank_data in comparative_report["rankings"]["overall"]:
        print(f"{rank_data['rank']}. {rank_data['model_id']}: {rank_data['average_score']:.3f}")

    print("\n\nMetric-wise Rankings:")
    print("-" * 60)
    for metric_name in sorted(all_metrics):
        print(f"\n{metric_name}:")
        for rank_data in comparative_report["rankings"][metric_name]:
            print(f"  {rank_data['rank']}. {rank_data['model_id']}: {rank_data['score']:.3f}")

    print(f"\n{'=' * 60}")
    print(f"Full comparative report saved to: {output_file}")
    print(f"{'=' * 60}\n")

