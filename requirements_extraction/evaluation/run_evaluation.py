if __name__ == "__main__":
    gemini_api_key = os.environ["GOOGLE_API_KEY"]

    # Run evaluation for all models
    print("\n" + "=" * 60)
    print("STARTING MODEL EVALUATION")
    print("=" * 60)

    evaluate_all_models(["llama3.1-8b-instruct",
                         "mistral-7B-instruct",
                         "qwen3-8b", "glm4-9b"],
                        gemini_api_key=gemini_api_key)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
