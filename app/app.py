from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api_schema import JobExtractionInput, JobExtractionOutput
from base_model import get_extractor_for

app = FastAPI()


@app.post("/extract", response_model=JobExtractionOutput)
async def extract_requirements(input: JobExtractionInput):
    try:
        model = get_extractor_for(input.modelId)
        requirements = model.process_text(input.inputText)
        return JobExtractionOutput(requirements=requirements)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
