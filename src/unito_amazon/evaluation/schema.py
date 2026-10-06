from pydantic import BaseModel, Field


class JudgeAssessment(BaseModel):
    relevance: int = Field(ge=1, le=5)
    respectfulness: int = Field(ge=1, le=5)
    persuasiveness: int = Field(ge=1, le=5)
    self_contained: int = Field(ge=1, le=5)
    conciseness: int = Field(ge=1, le=5)
    evidence_grounding: int | None = Field(default=None, ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1)


class JudgeResult(BaseModel):
    assessment: JudgeAssessment | None = None
    passed: bool | None = None
    raw_response: str
    parse_error: str | None = None
    attempts: int = Field(default=1, ge=1)
