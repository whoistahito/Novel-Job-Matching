import json
from pathlib import Path
from typing import List, Dict, Any

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from custom_llm import CustomNvidiaModel


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

        requirements = data.get("result", {}).get("requirements", {})
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


def create_evaluation_metrics(model: DeepEvalBaseLLM) -> List[GEval]:
    """Create evaluation metrics using G-Eval with a LLM as judge."""

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
        model=model,
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
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        model=model,
        threshold=0.7,
    )

    # Metric 3: Alignment - Did the model avoid extracting non-mandatory requirements?
    precision_metric = GEval(
        name="Alignment",
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
        model=model,
        threshold=0.7,
    )

    # Metric 4: Readability - Is the output well-organized and properly categorized?
    structure_metric = GEval(
        name="Readability",
        criteria="Evaluate whether the extracted requirements are properly categorized into skills, experiences, and qualifications, and whether the output is well-structured and easy to understand.",
        evaluation_steps=[
            "Review the categorization of requirements into skills, experiences, and qualifications",
            "Check if each requirement is placed in the appropriate category",
            "Evaluate the overall organization and readability of the output"
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        model=model,
        threshold=0.7,
    )

    return [correctness_metric, completeness_metric, precision_metric, structure_metric]


def run_evaluation(model_id: str, results_dir: str, api_key: str = None, sample_size: int = None):
    """Run evaluation for a specific model's results."""

    model = CustomNvidiaModel(
        api_key=api_key,
        model="openai/gpt-oss-120b",
        temperature=1.0
    )

    print(f"Evaluating model: {model_id}")

    if sample_size is None or sample_size <= 0:
        results = load_results_from_directory(results_dir)
    else:
        results = load_results_from_directory(results_dir)[:sample_size]

    if not results:
        print("No results found. Skipping evaluation.")
        return None

    test_cases = create_test_cases(results, model_id)
    metrics = create_evaluation_metrics(model)
    print(f"Created {len(metrics)} evaluation metrics:")
    for metric in metrics:
        print(f"  - {metric.name}")

    eval_results = evaluate(display_config=DisplayConfig(verbose_mode=False, print_results=False),
                            async_config=AsyncConfig(run_async=True, max_concurrent=4, throttle_value=2),
                            test_cases=test_cases, metrics=metrics)

    return eval_results


def save_evaluation_results(eval_results, model_id: str, output_dir: str = "response_evaluation"):
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

    # Average scores
    for metric_name, scores in metric_scores.items():
        results_summary["metrics_summary"][metric_name] = {
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "passed_count": sum(1 for score in scores if score >= 0.7)
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation results saved to: {output_file}")
    return results_summary
