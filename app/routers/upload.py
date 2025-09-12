from fastapi import APIRouter, HTTPException
from app.schemas.upload import UploadUrlRequest, UploadUrlResponse
from app.services.s3_service import S3Service
import structlog

logger = structlog.get_logger()
router = APIRouter(prefix="/upload-url", tags=["upload"])


@router.post("", response_model=UploadUrlResponse)
async def get_upload_url(request: UploadUrlRequest):
    """
    Generate a presigned URL for uploading a video file to S3.
    
    Args:
        request: Contains filename and content type
        
    Returns:
        Presigned upload URL and S3 file URL
    """
    try:
        s3_service = S3Service()
        upload_url, file_url, presigned_fields = s3_service.generate_presigned_upload_url(
            filename=request.filename,
            content_type=request.contentType
        )
        
        logger.info(
            "Generated upload URL",
            filename=request.filename,
            content_type=request.contentType
        )
        
        return UploadUrlResponse(
            uploadUrl=upload_url,
            fileUrl=file_url,
            contentType=request.contentType,
            contentDisposition="inline",
            uploadFields=presigned_fields
        )
        
    except Exception as e:
        logger.error(
            "Failed to generate upload URL",
            error=str(e),
            filename=request.filename
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate upload URL: {str(e)}"
        )
