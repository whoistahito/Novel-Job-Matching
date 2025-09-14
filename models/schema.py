from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Requirements(BaseModel):
    """Normalized requirements schema for job extraction outputs."""
    skills: List[str] = Field(default_factory=list, description="List of required skills")
    experience: List[str] = Field(default_factory=list, description="List of experience requirements")
    qualifications: List[str] = Field(default_factory=list, description="List of qualifications/certifications")
