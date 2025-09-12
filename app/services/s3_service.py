import boto3
import uuid
from datetime import datetime, timedelta
from typing import Tuple
from app.config import settings
import structlog

logger = structlog.get_logger()


class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        self.bucket_name = settings.s3_bucket_name

    def generate_presigned_upload_url(self, filename: str, content_type: str) -> Tuple[str, str, dict]:
        """
        Generate a presigned URL for uploading a file to S3.
        
        Args:
            filename: The original filename
            content_type: The MIME type of the file
            
        Returns:
            Tuple of (presigned_url, s3_file_url)
        """
        try:
            # Generate a unique key for the file
            file_extension = filename.split('.')[-1] if '.' in filename else 'mp4'
            unique_filename = f"swings/{uuid.uuid4()}.{file_extension}"
            
            # Validate and normalize content type for videos
            if not content_type or content_type == 'application/octet-stream':
                # Default to video/mp4 if no content type provided
                content_type = 'video/mp4'
            elif content_type.startswith('video/'):
                # Keep video content types as-is
                pass
            else:
                # Log warning for unexpected content types
                logger.warning(
                    "Unexpected content type for video upload",
                    filename=filename,
                    content_type=content_type
                )
            
            # Generate presigned POST for upload operation
            # This properly includes both ContentType and ContentDisposition in signed parameters
            presigned_post = self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=unique_filename,
                Fields={
                    'Content-Type': content_type,
                    'Content-Disposition': 'inline',
                },
                Conditions=[
                    {'Content-Type': content_type},
                    {'Content-Disposition': 'inline'},
                ],
                ExpiresIn=86400  # URL expires in 24 hours (longer for uploads)
            )
            
            # Extract the URL and fields for the response
            presigned_url = presigned_post['url']
            presigned_fields = presigned_post['fields']
            
            # Construct the S3 file URL (HTTPS format for web access)
            s3_file_url = f"https://{self.bucket_name}.s3.{settings.aws_region}.amazonaws.com/{unique_filename}"
            
            logger.info(
                "Generated presigned URL",
                filename=filename,
                s3_key=unique_filename,
                content_type=content_type
            )
            
            return presigned_url, s3_file_url, presigned_fields
            
        except Exception as e:
            logger.error(
                "Failed to generate presigned URL",
                error=str(e),
                filename=filename
            )
            raise Exception(f"Failed to generate presigned URL: {str(e)}")

    def generate_permanent_access_url(self, s3_url: str) -> str:
        """
        Generate a long-lived presigned URL for viewing a file.
        This URL will work for 7 days (604800 seconds).
        
        Args:
            s3_url: The S3 URL of the file (either s3:// or https:// format)
            
        Returns:
            Long-lived presigned URL for accessing the file
        """
        try:
            # Extract key from S3 URL
            if s3_url.startswith('s3://'):
                key = s3_url.split('/', 3)[3]  # Remove 's3://bucket-name/'
            elif s3_url.startswith('https://'):
                # Extract key from HTTPS URL, removing query parameters
                url_parts = s3_url.split('?')[0]  # Remove query parameters
                if f"{self.bucket_name}.s3.amazonaws.com" in url_parts:
                    key = url_parts.split(f"{self.bucket_name}.s3.amazonaws.com/")[1]
                elif f"{self.bucket_name}.s3.{settings.aws_region}.amazonaws.com" in url_parts:
                    key = url_parts.split(f"{self.bucket_name}.s3.{settings.aws_region}.amazonaws.com/")[1]
                else:
                    # Fallback: assume it's already a clean URL
                    return url_parts
            else:
                key = s3_url
                
            # Generate a long-lived presigned URL for GET operation (7 days)
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': key
                },
                ExpiresIn=604800  # 7 days
            )
            
            logger.info(
                "Generated long-lived access URL",
                s3_url=s3_url,
                presigned_url=presigned_url[:100] + "..."
            )
            
            return presigned_url
            
        except Exception as e:
            logger.error(
                "Failed to generate access URL",
                error=str(e),
                s3_url=s3_url
            )
            raise Exception(f"Failed to generate access URL: {str(e)}")

    def delete_file(self, s3_url: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            s3_url: The S3 URL of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract key from S3 URL
            if s3_url.startswith('s3://'):
                key = s3_url.split('/', 3)[3]  # Remove 's3://bucket-name/'
            else:
                key = s3_url
                
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            
            logger.info("Deleted file from S3", s3_url=s3_url, key=key)
            return True
            
        except Exception as e:
            logger.error(
                "Failed to delete file from S3",
                error=str(e),
                s3_url=s3_url
            )
            return False
