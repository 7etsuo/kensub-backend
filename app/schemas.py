from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class VideoJobResponse(BaseModel):
    id: str
    status: str
    progress: float
    original_filename: str
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class ClipInfo(BaseModel):
    title: str
    rating: float
    duration: float
    filename: str


class TranscriptWord(BaseModel):
    text: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[TranscriptWord]


class FullTranscript(BaseModel):
    segments: list[TranscriptSegment]
