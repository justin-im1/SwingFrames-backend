from pydantic import BaseModel


class SwingSaveResponse(BaseModel):
    swingId: int
    status: str
