"""
Evaluation framework for benchmarking LLMs at extracting job requirements.
Uses DeepEval with Gemini as the LLM judge.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def load_results_from_directory(results_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON result files from a directory."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Warning: Directory {results_dir} does not exist")
        return results

    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append({
                    "filename": json_file.name,
                    "data": data
                })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def create_test_cases(results: List[Dict[str, Any]], model_id: str) -> List[LLMTestCase]:
    """Create DeepEval test cases from result data."""
    test_cases = []

    for result in results:
        data = result["data"]

        # Extract the requirements
        requirements = data.get("result", {}).get("requirements", {})

        # Format the actual output as a readable string
        actual_output = json.dumps(requirements, indent=2)

        # Create test case
        test_case = LLMTestCase(
            input=data.get("inputText", ""),
            actual_output=actual_output,
            additional_metadata={
                "filename": result["filename"],
                "model_id": model_id,
                "processing_time": data.get("processingTimeInSeconds", 0)
            }
        )
        test_cases.append(test_case)

    return test_cases


def create_evaluation_metrics(gemini_model: GeminiModel) -> List[GEval]:
    """Create evaluation metrics using G-Eval with Gemini as judge."""

    # Metric 1: Correctness - Are the extracted requirements accurate?
    correctness_metric = GEval(
        name="Correctness",
        criteria="Evaluate whether the extracted job requirements are accurate and actually present in the input text. Check if the model correctly identified mandatory requirements and avoided adding information that wasn't in the original text.",
        evaluation_steps=[
            "Read the input job description carefully",
            "Examine each extracted requirement (skills, experiences, qualifications)",
            "Verify that each requirement is explicitly stated in the input text",
            "Check that no requirements were hallucinated or fabricated",
            "Assess whether the categorization (skills/experiences/qualifications) is appropriate"
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        model=gemini_model,
        threshold=0.7,
    )

    # Metric 2: Completeness - Did the model extract all mandatory requirements?
    completeness_metric = GEval(
        name="Completeness",
        criteria="Evaluate whether all mandatory job requirements from the input text were extracted. The model should capture all 'must-have', 'required', and 'essential' requirements without missing important details.",
        evaluation_steps=[
            "Identify all mandatory requirements mentioned in the input job description",
            "Look for keywords like 'required', 'must have', 'essential', 'necessary'",
            "Check if the actual output includes all these mandatory requirements",
            "Assess whether any critical requirements were omitted",
            "Verify that the extraction is comprehensive"
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        model=gemini_model,
        threshold=0.7,
    )

    # Metric 3: Precision - Did the model avoid extracting non-mandatory requirements?
    precision_metric = GEval(
        name="Precision",
        criteria="Evaluate whether the model correctly avoided extracting 'nice-to-have', 'preferred', or 'bonus' requirements. Only mandatory requirements should be included.",
        evaluation_steps=[
            "Identify any 'preferred', 'nice-to-have', 'bonus', or optional requirements in the input",
            "Check if the actual output incorrectly includes any of these optional items",
            "Verify that the model focused only on mandatory requirements",
            "Assess the model's ability to distinguish between required and optional qualifications"
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        model=gemini_model,
        threshold=0.7,
    )

    # Metric 4: Structure Quality - Is the output well-organized and properly categorized?
    structure_metric = GEval(
        name="Structure Quality",
        criteria="Evaluate whether the extracted requirements are properly categorized into skills, experiences, and qualifications, and whether the output is well-structured and easy to understand.",
        evaluation_steps=[
            "Review the categorization of requirements into skills, experiences, and qualifications",
            "Check if each requirement is placed in the appropriate category",
            "Assess whether the requirements are specific and concise",
            "Verify that each requirement is clearly stated without ambiguity",
            "Evaluate the overall organization and readability of the output"
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        model=gemini_model,
        threshold=0.7,
    )

    return [correctness_metric, completeness_metric, precision_metric, structure_metric]


def run_evaluation(model_id: str, results_dir: str, gemini_api_key: str = None, sample_size: int = None):
    """Run evaluation for a specific model's results."""

    # Set up Gemini API key if provided
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key

    # Initialize Gemini model
    gemini_model = GeminiModel(
        model_name="gemini-2.5-flash-lite",
        api_key=gemini_api_key
    )

    # Load results
    print(f"\n{'=' * 60}")
    print(f"Evaluating model: {model_id}")
    print(f"Results directory: {results_dir}")
    print(f"{'=' * 60}\n")

    if sample_size is None or sample_size <= 0:
        results = load_results_from_directory(results_dir)[:sample_size]
    else:
        results = load_results_from_directory(results_dir)

    print(f"Loaded {len(results)} result files")

    if not results:
        print("No results found. Skipping evaluation.")
        return None

    # Create test cases
    test_cases = create_test_cases(results, model_id)
    print(f"Created {len(test_cases)} test cases")

    # Create evaluation metrics
    metrics = create_evaluation_metrics(gemini_model)
    print(f"Created {len(metrics)} evaluation metrics:")
    for metric in metrics:
        print(f"  - {metric.name}")

    eval_results = evaluate(async_config=AsyncConfig(run_async=True, max_concurrent=1, throttle_value=20),
                            test_cases=test_cases, metrics=metrics)

    return eval_results


def save_evaluation_results(eval_results, model_id: str, output_dir: str = "../evaluation_results"):
    """Save evaluation results to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_file = output_path / f"{model_id}_evaluation.json"

    # Extract results summary
    results_summary = {
        "model_id": model_id,
        "total_test_cases": len(eval_results.test_results),
        "metrics_summary": {},
        "test_results": []
    }

    # Aggregate metric scores
    metric_scores = {}
    for test_result in eval_results.test_results:
        for metric_result in test_result.metrics_data:
            metric_name = metric_result.name
            if metric_name not in metric_scores:
                metric_scores[metric_name] = []
            metric_scores[metric_name].append(metric_result.score)

        # Add individual test result
        results_summary["test_results"].append({
            "filename": test_result.additional_metadata.get("filename", "unknown"),
            "processing_time": test_result.additional_metadata.get("processing_time", 0),
            "metrics": {
                metric_result.name: {
                    "score": metric_result.score,
                    "success": metric_result.success,
                    "reason": metric_result.reason
                }
                for metric_result in test_result.metrics_data
            }
        })

    # Calculate average scores
    for metric_name, scores in metric_scores.items():
        results_summary["metrics_summary"][metric_name] = {
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "passed_count": sum(1 for score in scores if score >= 0.7)
        }

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation results saved to: {output_file}")
    return results_summary


def print_evaluation_summary(results_summary: Dict[str, Any]):
    """Print a formatted summary of evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"EVALUATION SUMMARY FOR: {results_summary['model_id']}")
    print(f"{'=' * 60}\n")

    print(f"Total test cases: {results_summary['total_test_cases']}\n")

    print("Metrics Performance:")
    print("-" * 60)
    for metric_name, metric_data in results_summary["metrics_summary"].items():
        avg_score = metric_data["average_score"]
        passed = metric_data["passed_count"]
        total = results_summary["total_test_cases"]

        print(f"\n{metric_name}:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Min Score: {metric_data['min_score']:.3f}")
        print(f"  Max Score: {metric_data['max_score']:.3f}")
        print(f"  Passed (≥0.7): {passed}/{total} ({100 * passed / total:.1f}%)")

    print(f"\n{'=' * 60}\n")
