from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from app.schemas.swing import SwingComparison


class ComparisonBase(BaseModel):
    swing1_id: int
    swing2_id: int
    notes: Optional[str] = None


class ComparisonCreate(ComparisonBase):
    pass


class ComparisonResponse(ComparisonBase):
    id: int
    user_id: str  # Clerk user ID
    created_at: datetime
    swing1: SwingComparison
    swing2: SwingComparison

    class Config:
        from_attributes = True
