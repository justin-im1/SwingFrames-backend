from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class SwingBase(BaseModel):
    file_url: str
    tag: Optional[str] = None


class SwingCreate(SwingBase):
    pass


class SwingResponse(SwingBase):
    id: int
    user_id: str  # Clerk user ID
    created_at: datetime

    class Config:
        from_attributes = True


class SwingComparison(BaseModel):
    id: int
    file_url: str
    tag: Optional[str]
    created_at: datetime
    user_id: str  # Clerk user ID

    class Config:
        from_attributes = True
