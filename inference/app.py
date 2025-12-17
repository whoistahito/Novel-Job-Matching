from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api_schema import JobExtractionInput, SimilarityScore
from base_model import get_extractor_for
from similarity_search import compute_similarity

app = FastAPI()


@app.post("/extract", response_model=SimilarityScore)
async def extract_requirements(input: JobExtractionInput):
    try:
        model = get_extractor_for(input.modelId)
        requirements = model.process_text(input.inputText)
        return compute_similarity(input.userProfile, requirements)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
