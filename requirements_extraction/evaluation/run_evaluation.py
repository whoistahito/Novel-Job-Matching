from batch_evaluation import evaluate_all_models

gemini_api_key = os.environ["EXTERNAL_LLM_API_KEY"]

# Run evaluation for all models
print("\n" + "=" * 60)
print("STARTING MODEL EVALUATION")
print("=" * 60)

evaluate_all_models(["deepseek-v3.1-terminus"],
                    gemini_api_key=gemini_api_key)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
