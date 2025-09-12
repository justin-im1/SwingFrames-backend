from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Swing(Base):
    __tablename__ = "swings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)  # Clerk user ID
    file_url = Column(String, nullable=False)
    tag = Column(String, nullable=True)  # e.g., "slice", "hook", "straight"
    snapshot_urls = Column(JSON, nullable=True)  # Dict of event -> snapshot URL
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    user = relationship("User", back_populates="swings")
