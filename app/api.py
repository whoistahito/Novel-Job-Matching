from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.adapters.echo_adapter import EchoAdapter
from app.adapters.glm4_adapter import GLM4Adapter
from app.adapters.glm4_z1_adapter import GLM4Z1Adapter
from app.adapters.llama_nemotron_adapter import LlamaNemotronAdapter
from app.adapters.mistral_adapter import MistralAdapter
from app.adapters.qwen_adapter import QwenAdapter
from app.core.contracts import (
    ErrorDetail,
    ErrorResponse,
    InferenceRequest,
    InferenceResponse,
)
from app.core.registry import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: register adapters
    registry.register(EchoAdapter())
    registry.register(GLM4Adapter())
    registry.register(GLM4Z1Adapter())
    registry.register(LlamaNemotronAdapter())
    registry.register(MistralAdapter())
    registry.register(QwenAdapter())
    yield
    # Shutdown hook: nothing yet


app = FastAPI(title="Unified Inference API", version="0.1.0", lifespan=lifespan)


@app.get("/models")
async def list_models() -> dict[str, list[str]]:
    return {"models": registry.list()}


@app.post("/inference", response_model=InferenceResponse, responses={
    404: {"model": ErrorResponse},
    400: {"model": ErrorResponse},
    501: {"model": ErrorResponse},
    422: {"description": "Validation Error"},
})
async def inference(body: InferenceRequest):
    try:
        adapter = registry.get(body.model)
    except KeyError:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(code="model_not_found", message=f"Unknown model: {body.model}")
            ).model_dump(),
        )

    if body.stream:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="stream_not_supported",
                    message=f"Streaming not supported for model: {body.model}",
                )
            ).model_dump(),
        )

    # Validate and run prediction
    data = adapter.input_model(**body.input)
    try:
        out = await adapter.predict(data, body.params)
    except NotImplementedError as e:
        return JSONResponse(
            status_code=501,
            content=ErrorResponse(
                error=ErrorDetail(code="not_implemented", message=str(e) or "Not implemented")
            ).model_dump(),
        )

    # Ensure the output is validated and serializable
    out_validated = adapter.output_model.model_validate(out)

    return InferenceResponse(model=body.model, output=out_validated.model_dump())
