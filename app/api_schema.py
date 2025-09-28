from pydantic import BaseModel, Field


class Requirements(BaseModel):
    skills: list[str] = Field(default_factory=list)
    experiences: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(default_factory=list)


class JobExtractionInput(BaseModel):
    inputText: str = Field(..., description="Text to extract job requirements from")
    modelId: str = Field(..., description="LLM model id to use")


class JobExtractionOutput(BaseModel):
    requirements: Requirements
