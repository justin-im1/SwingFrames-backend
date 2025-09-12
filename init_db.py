#!/usr/bin/env python3
"""
Initialize the database with tables and sample data.
"""

import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.database import engine, SessionLocal
from app.models.user import User
from app.models.swing import Swing
from app.models.comparison import Comparison
import structlog

logger = structlog.get_logger()


def init_database():
    """Initialize the database with tables."""
    try:
        # Import all models to ensure they're registered
        from app.models import user, swing, comparison
        
        # Create all tables
        from app.database import Base
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database tables created successfully")
        
        # Create sample data
        create_sample_data()
        
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


def create_sample_data():
    """Create sample data for testing."""
    db = SessionLocal()
    
    try:
        # Check if sample data already exists
        existing_user = db.query(User).filter(User.email == "demo@golfswing.com").first()
        if existing_user:
            logger.info("Sample data already exists, skipping creation")
            return
        
        # Create sample user
        user = User(
            name="Demo User",
            email="demo@golfswing.com"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Create sample swings
        swing1 = Swing(
            user_id=user.id,
            file_url="s3://demo-bucket/swing1.mp4",
            tag="slice"
        )
        
        swing2 = Swing(
            user_id=user.id,
            file_url="s3://demo-bucket/swing2.mp4",
            tag="hook"
        )
        
        db.add(swing1)
        db.add(swing2)
        db.commit()
        
        logger.info(
            "Sample data created successfully",
            user_id=user.id,
            swing_count=2
        )
        
    except Exception as e:
        db.rollback()
        logger.error("Failed to create sample data", error=str(e))
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_database()
    print("Database initialized successfully!")
