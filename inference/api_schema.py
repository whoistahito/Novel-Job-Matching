from typing import List

from pydantic import BaseModel, Field


class SkillExperienceBase(BaseModel):
    """Base class for skill and experience data structures."""
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


class Requirements(SkillExperienceBase):
    """Normalized requirements schema for job extraction outputs."""
    pass


class UserProfile(SkillExperienceBase):
    """User profile with skills, experiences, and qualifications."""
    pass


class JobExtractionInput(BaseModel):
    inputText: str = Field(...,
                           description="Text to extract job requirements from"
                           )
    modelId: str = Field(...,
                         description="LLM model id to use"
                         )
    userProfile: UserProfile = Field(...,
                                     description="User profile data"
                                     )


class JobExtractionOutput(BaseModel):
    requirements: Requirements
