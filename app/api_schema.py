from pydantic import BaseModel, Field


class Requirements(BaseModel):
    """Normalized requirements schema for job extraction outputs."""
    skills: List[str] = Field(
        default_factory=list,
        description="List of required skills"
    )
    experiences: List[str] = Field(
        default_factory=list,
        description="List of experience requirements"
    )
    qualifications: List[str] = Field(
        default_factory=list,
        description="List of qualifications/certifications"
    )


class JobExtractionInput(BaseModel):
    inputText: str = Field(...,
                           description="Text to extract job requirements from"
                           )
    modelId: str = Field(...,
                         description="LLM model id to use"
                         )


class JobExtractionOutput(BaseModel):
    requirements: Requirements
