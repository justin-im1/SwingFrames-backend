from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.swing import Swing
from app.models.user import User
from app.schemas.swing import SwingCreate, SwingResponse
from app.schemas.response import SwingSaveResponse
from app.middleware.auth import get_current_user_id, verify_clerk_token
from app.services.s3_service import S3Service
from app.services.snapshot_generator import SnapshotGenerator
from app.services.pose_analysis import extract_landmarks, detect_swing_events
import structlog
import tempfile
import os
from pathlib import Path

security = HTTPBearer()

logger = structlog.get_logger()
router = APIRouter(prefix="/swings", tags=["swings"])


@router.post("", response_model=SwingSaveResponse)
async def create_swing(
    swing_data: SwingCreate, 
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Save swing metadata to the database.
    
    Args:
        swing_data: Swing data including file_url and tag
        current_user_id: Current user ID from JWT token
        db: Database session
        
    Returns:
        Swing ID and status
    """
    try:
        # Create or get user record
        user = db.query(User).filter(User.id == current_user_id).first()
        if not user:
            # Create user record from JWT token data
            user_data = await verify_clerk_token(credentials)
            
            # Debug: Log what we got from the token
            logger.info(
                "Creating user from JWT token",
                user_id=current_user_id,
                token_data=user_data
            )
            
            # Use user_id as email if email is not provided (Clerk fallback)
            email = user_data.get("email") or f"{current_user_id}@clerk.local"
            user = User(
                id=current_user_id,
                email=email,
                first_name=user_data.get("first_name"),
                last_name=user_data.get("last_name")
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        
        # Convert the file URL to a permanent access URL before storing
        s3_service = S3Service()
        permanent_url = s3_service.generate_permanent_access_url(swing_data.file_url)
        
        # Create new swing record
        swing = Swing(
            user_id=current_user_id,
            file_url=permanent_url,
            tag=swing_data.tag
        )
        
        db.add(swing)
        db.commit()
        db.refresh(swing)
        
        logger.info(
            "Swing saved successfully",
            swing_id=swing.id,
            user_id=current_user_id,
            tag=swing_data.tag
        )
        
        return SwingSaveResponse(
            swingId=swing.id,
            status="saved"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(
            "Failed to save swing",
            error=str(e),
            user_id=current_user_id
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save swing: {str(e)}"
        )


@router.get("", response_model=List[SwingResponse])
async def get_user_swings(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Fetch all swings for the current user.
    
    Args:
        current_user_id: Current user ID from JWT token
        db: Database session
        
    Returns:
        List of swings for the user
    """
    try:
        # Get all swings for the user using raw SQL to avoid relationship issues
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id, file_url, tag, snapshot_urls, created_at FROM swings WHERE user_id = :user_id"),
            {"user_id": current_user_id}
        )
        swings_data = result.fetchall()
        
        # Convert to SwingResponse objects
        s3_service = S3Service()
        swings = []
        for row in swings_data:
            # Convert S3 URL to permanent access URL
            permanent_url = s3_service.generate_permanent_access_url(row.file_url)
            
            swing_response = SwingResponse(
                id=row.id,
                user_id=row.user_id,
                file_url=permanent_url,
                tag=row.tag,
                snapshot_urls=row.snapshot_urls,
                created_at=row.created_at
            )
            swings.append(swing_response)
        
        logger.info(
            "Retrieved user swings",
            user_id=current_user_id,
            swing_count=len(swings)
        )
        
        return swings
        
    except Exception as e:
        logger.error(
            "Failed to retrieve user swings",
            error=str(e),
            user_id=current_user_id
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve swings: {str(e)}"
        )


@router.get("/{swing_id}/access-url")
async def get_swing_access_url(
    swing_id: int,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Get a permanent access URL for a swing video.
    
    Args:
        swing_id: The ID of the swing
        current_user_id: Current user ID from JWT token
        db: Database session
        
    Returns:
        Permanent access URL for the swing video
    """
    try:
        # Get the swing and verify ownership using raw SQL
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id, file_url, tag, snapshot_urls, created_at FROM swings WHERE id = :swing_id AND user_id = :user_id"),
            {"swing_id": swing_id, "user_id": current_user_id}
        )
        swing_data = result.fetchone()
        
        if not swing_data:
            raise HTTPException(
                status_code=404,
                detail="Swing not found or access denied"
            )
        
        # Generate permanent access URL
        s3_service = S3Service()
        access_url = s3_service.generate_permanent_access_url(swing_data.file_url)
        
        logger.info(
            "Generated access URL for swing",
            swing_id=swing_id,
            user_id=current_user_id
        )
        
        return {
            "swingId": swing_id,
            "accessUrl": access_url,
            "fileUrl": swing_data.file_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to generate access URL",
            error=str(e),
            swing_id=swing_id,
            user_id=current_user_id
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate access URL: {str(e)}"
        )


@router.delete("/{swing_id}")
async def delete_swing(
    swing_id: int,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Delete a swing and its associated S3 file.
    
    Args:
        swing_id: The ID of the swing to delete
        current_user_id: Current user ID from JWT token
        db: Database session
        
    Returns:
        Success message
    """
    try:
        # Get the swing and verify ownership using raw SQL
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id, file_url, tag, snapshot_urls, created_at FROM swings WHERE id = :swing_id AND user_id = :user_id"),
            {"swing_id": swing_id, "user_id": current_user_id}
        )
        swing_data = result.fetchone()
        
        if not swing_data:
            raise HTTPException(
                status_code=404,
                detail="Swing not found or access denied"
            )
        
        # Extract S3 key from the file URL for deletion
        s3_key = None
        if swing_data.file_url:
            if 's3.amazonaws.com/' in swing_data.file_url:
                s3_key = swing_data.file_url.split('s3.amazonaws.com/')[-1]
            elif 's3.' in swing_data.file_url and '.amazonaws.com/' in swing_data.file_url:
                s3_key = swing_data.file_url.split('.amazonaws.com/')[-1]
        
        # Delete from S3 if we have a valid key
        if s3_key:
            try:
                s3_service = S3Service()
                s3_service.s3_client.delete_object(
                    Bucket=s3_service.bucket_name,
                    Key=s3_key
                )
                logger.info(
                    "Deleted S3 file for swing",
                    swing_id=swing_id,
                    s3_key=s3_key
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete S3 file for swing",
                    swing_id=swing_id,
                    s3_key=s3_key,
                    error=str(e)
                )
                # Continue with database deletion even if S3 deletion fails
        
        # Delete from database using raw SQL
        db.execute(
            text("DELETE FROM swings WHERE id = :swing_id AND user_id = :user_id"),
            {"swing_id": swing_id, "user_id": current_user_id}
        )
        db.commit()
        
        logger.info(
            "Swing deleted successfully",
            swing_id=swing_id,
            user_id=current_user_id
        )
        
        return {
            "message": "Swing deleted successfully",
            "swingId": swing_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(
            "Failed to delete swing",
            error=str(e),
            swing_id=swing_id,
            user_id=current_user_id
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete swing: {str(e)}"
        )


@router.get("/{swing_id}/snapshots/presigned-urls")
async def get_swing_snapshot_presigned_urls(
    swing_id: int,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Get presigned URLs for swing snapshots.
    
    Args:
        swing_id: The ID of the swing
        current_user_id: Current user ID from JWT token
        db: Database session
        
    Returns:
        Presigned URLs for the swing snapshots
    """
    try:
        # Get the swing and verify ownership using raw SQL
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id, file_url, tag, snapshot_urls, created_at FROM swings WHERE id = :swing_id AND user_id = :user_id"),
            {"swing_id": swing_id, "user_id": current_user_id}
        )
        swing_data = result.fetchone()
        
        if not swing_data:
            raise HTTPException(
                status_code=404,
                detail="Swing not found or access denied"
            )
        
        # Check if snapshots exist
        if not swing_data.snapshot_urls:
            logger.info(
                "No snapshots found for swing",
                swing_id=swing_id,
                user_id=current_user_id
            )
            
            return {
                "swing_id": swing_id,
                "snapshots": {},
                "status": "not_generated"
            }
        
        # Generate presigned URLs for each snapshot
        s3_service = S3Service()
        presigned_snapshots = {}
        
        import json
        snapshots = json.loads(swing_data.snapshot_urls) if isinstance(swing_data.snapshot_urls, str) else swing_data.snapshot_urls
        
        for event_name, snapshot_url in snapshots.items():
            # Extract S3 key from the snapshot URL
            if 's3.amazonaws.com/' in snapshot_url:
                s3_key = snapshot_url.split('s3.amazonaws.com/')[-1]
            elif 's3.' in snapshot_url and '.amazonaws.com/' in snapshot_url:
                s3_key = snapshot_url.split('.amazonaws.com/')[-1]
            else:
                continue
            
            # Generate presigned URL for viewing the snapshot
            try:
                presigned_url = s3_service.s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': s3_service.bucket_name,
                        'Key': s3_key,
                        'ResponseContentType': 'image/jpeg',
                        'ResponseContentDisposition': 'inline'
                    },
                    ExpiresIn=3600  # 1 hour
                )
                presigned_snapshots[event_name] = presigned_url
                
            except Exception as e:
                logger.error(
                    "Failed to generate presigned URL for snapshot",
                    error=str(e),
                    swing_id=swing_id,
                    event_name=event_name,
                    s3_key=s3_key
                )
                # Fall back to the original URL
                presigned_snapshots[event_name] = snapshot_url
        
        logger.info(
            "Generated presigned URLs for snapshots",
            swing_id=swing_id,
            user_id=current_user_id,
            snapshot_count=len(presigned_snapshots)
        )
        
        return {
            "swing_id": swing_id,
            "snapshots": presigned_snapshots,
            "status": "available"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to generate presigned URLs for snapshots",
            error=str(e),
            swing_id=swing_id,
            user_id=current_user_id
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate presigned URLs: {str(e)}"
        )


@router.get("/{swing_id}/snapshots")
async def get_swing_snapshots(
    swing_id: int,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Get snapshot URLs for a swing.
    
    Args:
        swing_id: The ID of the swing
        current_user_id: Current user ID from JWT token
        db: Database session
        
    Returns:
        Snapshot URLs for the swing
    """
    try:
        # Get the swing and verify ownership using raw SQL
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id, file_url, tag, snapshot_urls, created_at FROM swings WHERE id = :swing_id AND user_id = :user_id"),
            {"swing_id": swing_id, "user_id": current_user_id}
        )
        swing_data = result.fetchone()
        
        if not swing_data:
            raise HTTPException(
                status_code=404,
                detail="Swing not found or access denied"
            )
        
        # Check if snapshots already exist
        if swing_data.snapshot_urls:
            logger.info(
                "Retrieved existing swing snapshots",
                swing_id=swing_id,
                user_id=current_user_id,
                snapshot_count=len(swing_data.snapshot_urls) if isinstance(swing_data.snapshot_urls, dict) else 0
            )
            
            return {
                "swing_id": swing_id,
                "snapshots": swing_data.snapshot_urls,
                "status": "available"
            }
        else:
            logger.info(
                "No snapshots found for swing",
                swing_id=swing_id,
                user_id=current_user_id
            )
            
            return {
                "swing_id": swing_id,
                "snapshots": {},
                "status": "not_generated"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve snapshots",
            error=str(e),
            swing_id=swing_id,
            user_id=current_user_id
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve snapshots: {str(e)}"
        )


@router.post("/{swing_id}/snapshots/generate")
async def generate_swing_snapshots(
    swing_id: int,
    background_tasks: BackgroundTasks,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Generate snapshots for a swing (placeholder endpoint).
    
    Args:
        swing_id: The ID of the swing
        current_user_id: Current user ID from JWT token
        db: Database session
        
    Returns:
        Generation status
    """
    try:
        # Get the swing and verify ownership using raw SQL
        from sqlalchemy import text
        result = db.execute(
            text("SELECT id, user_id, file_url, tag, snapshot_urls, created_at FROM swings WHERE id = :swing_id AND user_id = :user_id"),
            {"swing_id": swing_id, "user_id": current_user_id}
        )
        swing_data = result.fetchone()
        
        if not swing_data:
            raise HTTPException(
                status_code=404,
                detail="Swing not found or access denied"
            )
        
        # Check if snapshots already exist
        if swing_data.snapshot_urls:
            logger.info(
                "Snapshots already exist for swing",
                swing_id=swing_id,
                user_id=current_user_id
            )
            return {
                "message": "Snapshots already exist",
                "swing_id": swing_id,
                "snapshot_urls": swing_data.snapshot_urls,
                "status": "already_exists"
            }
        
        # Start background task for snapshot generation
        task_id = f"snapshot_gen_{swing_id}_{current_user_id}"
        background_tasks.add_task(
            _generate_snapshots_background,
            swing_id=swing_id,
            user_id=current_user_id,
            file_url=swing_data.file_url
        )
        
        logger.info(
            "Started snapshot generation task",
            swing_id=swing_id,
            user_id=current_user_id,
            task_id=task_id
        )
        
        return {
            "message": "Snapshot generation started",
            "swing_id": swing_id,
            "task_id": task_id,
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to start snapshot generation",
            error=str(e),
            swing_id=swing_id,
            user_id=current_user_id
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start snapshot generation: {str(e)}"
        )


async def _generate_snapshots_background(swing_id: int, user_id: str, file_url: str):
    """
    Background task to generate snapshots and upload to S3.
    
    Args:
        swing_id: The swing ID
        user_id: The user ID
        file_url: The swing video file URL
    """
    from app.database import get_db
    
    logger.info(
        "Starting background snapshot generation",
        swing_id=swing_id,
        user_id=user_id
    )
    
    try:
        # Get database session
        db = next(get_db())
        
        # Initialize services
        s3_service = S3Service()
        snapshot_generator = SnapshotGenerator()
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Download video from S3 to temporary location
            # Extract S3 key from file URL (strip query parameters)
            if 's3.amazonaws.com/' in file_url:
                video_key = file_url.split('s3.amazonaws.com/')[-1]
            elif 's3.' in file_url and '.amazonaws.com/' in file_url:
                video_key = file_url.split('.amazonaws.com/')[-1]
            else:
                raise ValueError(f"Invalid S3 URL format: {file_url}")
            
            # Remove query parameters from S3 key
            if '?' in video_key:
                video_key = video_key.split('?')[0]
            
            temp_video_path = temp_dir_path / "swing_video.mp4"
            
            logger.info(
                "Downloading video from S3",
                swing_id=swing_id,
                s3_key=video_key
            )
            
            s3_service.s3_client.download_file(
                s3_service.bucket_name,
                video_key,
                str(temp_video_path)
            )
            
            # Generate snapshots
            logger.info(
                "Generating snapshots",
                swing_id=swing_id,
                video_path=str(temp_video_path)
            )
            
            snapshots = snapshot_generator.generate_swing_snapshots(str(temp_video_path))
            
            # Upload snapshots to S3
            snapshot_urls = {}
            for event_name, local_path in snapshots.items():
                # Generate S3 key for snapshot
                snapshot_key = f"snapshots/{swing_id}/{event_name}.jpg"
                
                # Upload to S3
                s3_service.s3_client.upload_file(
                    local_path,
                    s3_service.bucket_name,
                    snapshot_key,
                    ExtraArgs={
                        'ContentType': 'image/jpeg',
                        'ContentDisposition': 'inline'
                    }
                )
                
                # Generate permanent access URL
                from app.config import settings
                snapshot_url = f"https://{s3_service.bucket_name}.s3.{settings.aws_region}.amazonaws.com/{snapshot_key}"
                snapshot_urls[event_name] = snapshot_url
                
                logger.info(
                    "Uploaded snapshot to S3",
                    swing_id=swing_id,
                    event_name=event_name,
                    s3_key=snapshot_key
                )
            
            # Update swing record with snapshot URLs using raw SQL
            import json
            from sqlalchemy import text
            db.execute(
                text("UPDATE swings SET snapshot_urls = :snapshot_urls WHERE id = :swing_id"),
                {"snapshot_urls": json.dumps(snapshot_urls), "swing_id": swing_id}
            )
            db.commit()
            
            logger.info(
                "Updated swing with snapshot URLs",
                swing_id=swing_id,
                snapshot_count=len(snapshot_urls)
            )
        
    except Exception as e:
        logger.error(
            "Background snapshot generation failed",
            error=str(e),
            swing_id=swing_id,
            user_id=user_id
        )
        
        # Update swing record to indicate failure using raw SQL
        try:
            import json
            from sqlalchemy import text
            db.execute(
                text("UPDATE swings SET snapshot_urls = :error_data WHERE id = :swing_id"),
                {"error_data": json.dumps({"error": str(e)}), "swing_id": swing_id}
            )
            db.commit()
        except:
            pass  # Don't fail the background task if we can't update the database
    
    finally:
        db.close()
