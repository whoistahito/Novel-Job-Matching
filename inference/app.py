from fastapi import FastAPI, HTTPException, Response, status

from .api_schema import JobExtractionInput, JobMatchingResponse
from .base_model import get_extractor_for
from .similarity_search import compute_similarity

app = FastAPI()


@app.post("/extract", response_model=JobMatchingResponse)
async def extract_requirements(input: JobExtractionInput,
                               response: Response,
                               ):
    try:
        model = get_extractor_for(input.modelId)
        requirements = model.process_text(input.inputText)
        score = compute_similarity(input.userProfile, requirements)
        response.status_code = status.HTTP_200_OK
        return JobMatchingResponse(jobRequirements=requirements,
                                   userProfile=input.userProfile,
                                   similarityScore=score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
