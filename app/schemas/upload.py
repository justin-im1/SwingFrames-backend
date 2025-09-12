from pydantic import BaseModel


class UploadUrlRequest(BaseModel):
    filename: str
    contentType: str


class UploadUrlResponse(BaseModel):
    uploadUrl: str
    fileUrl: str
    contentType: str
    contentDisposition: str = "inline"
    uploadFields: dict  # Fields for presigned POST
