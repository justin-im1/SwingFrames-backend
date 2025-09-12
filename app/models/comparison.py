from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Comparison(Base):
    __tablename__ = "comparisons"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)  # Clerk user ID
    swing1_id = Column(Integer, ForeignKey("swings.id"), nullable=False)
    swing2_id = Column(Integer, ForeignKey("swings.id"), nullable=False)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="comparisons")
    swing1 = relationship("Swing", foreign_keys=[swing1_id])
    swing2 = relationship("Swing", foreign_keys=[swing2_id])
