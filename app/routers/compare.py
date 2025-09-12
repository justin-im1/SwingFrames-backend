from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.swing import Swing
from app.middleware.auth import get_current_user_id
from app.schemas.swing import SwingComparison
import structlog

logger = structlog.get_logger()
router = APIRouter(prefix="/compare", tags=["compare"])


@router.get("/{swing_id1}/{swing_id2}")
async def compare_swings(
    swing_id1: int, 
    swing_id2: int, 
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Fetch metadata for two swings to enable comparison.
    
    Args:
        swing_id1: ID of the first swing
        swing_id2: ID of the second swing
        current_user_id: Current user ID from JWT token
        db: Database session
        
    Returns:
        Metadata for both swings including URLs and timestamps
    """
    try:
        # Fetch both swings
        swing1 = db.query(Swing).filter(Swing.id == swing_id1).first()
        swing2 = db.query(Swing).filter(Swing.id == swing_id2).first()
        
        if not swing1:
            raise HTTPException(status_code=404, detail=f"Swing {swing_id1} not found")
        if not swing2:
            raise HTTPException(status_code=404, detail=f"Swing {swing_id2} not found")
        
        # Verify both swings belong to the current user
        if swing1.user_id != current_user_id:
            raise HTTPException(
                status_code=403, 
                detail="Access denied: Swing does not belong to current user"
            )
        if swing2.user_id != current_user_id:
            raise HTTPException(
                status_code=403, 
                detail="Access denied: Swing does not belong to current user"
            )
        
        logger.info(
            "Retrieved swing comparison data",
            swing1_id=swing_id1,
            swing2_id=swing_id2,
            user_id=current_user_id
        )
        
        return {
            "swing1": SwingComparison.from_orm(swing1),
            "swing2": SwingComparison.from_orm(swing2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve swing comparison data",
            error=str(e),
            swing1_id=swing_id1,
            swing2_id=swing_id2
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve comparison data: {str(e)}"
        )
