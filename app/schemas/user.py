from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.schemas.swing import SwingResponse
    from app.schemas.comparison import ComparisonResponse


class UserBase(BaseModel):
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserCreate(UserBase):
    pass


class UserResponse(UserBase):
    id: str  # Clerk user ID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserWithSwings(UserResponse):
    swings: List["SwingResponse"] = []


class UserWithComparisons(UserResponse):
    comparisons: List["ComparisonResponse"] = []
