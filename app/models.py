from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, Boolean, Integer
from .database import Base


class VideoJob(Base):
    __tablename__ = "video_jobs"

    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    status = Column(String)  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)
    original_filename = Column(String)
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    is_deleted = Column(Boolean, default=False)

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "original_filename": self.original_filename,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message
        }
