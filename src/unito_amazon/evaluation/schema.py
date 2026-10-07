from typing import Literal

from pydantic import BaseModel, Field


PairwiseWinner = Literal["response_1", "response_2", "tie"]


class PairwiseJudgeResult(BaseModel):
    winner: PairwiseWinner | None = None
    response_1_score: float | None = None
    response_2_score: float | None = None
    raw_response: str
    parse_error: str | None = None
    attempts: int = Field(default=1, ge=1)
