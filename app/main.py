import json
import shutil
import aiofiles
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.orm import Session
import aioredis

from .config import settings
from .database import get_db, engine
from .models import Base, VideoJob
from .schemas import VideoJobResponse, ClipInfo, FullTranscript
from .tasks import process_video

app = FastAPI(title="Video Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    redis = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True
    )
    await FastAPILimiter.init(redis)

@app.post("/videos/", response_model=VideoJobResponse, dependencies=[
    Depends(RateLimiter(times=10, minutes=1))
])
async def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a video for processing.
    No API key required.
    """
    try:
        job_id = str(uuid4())
        upload_dir = settings.UPLOAD_DIR / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / "video.mp4"

        file_size = 0
        async with aiofiles.open(file_path, 'wb') as out_file:
            while chunk := await file.read(settings.CHUNK_SIZE):
                file_size += len(chunk)
                if file_size > settings.MAX_UPLOAD_SIZE:
                    await out_file.close()
                    shutil.rmtree(upload_dir)
                    raise HTTPException(
                        status_code=413,
                        detail="File too large"
                    )
                await out_file.write(chunk)

        db_job = VideoJob(
            id=job_id,
            user_id="anonymous",
            status="pending",
            original_filename=file.filename,
            file_size=file_size,
            created_at=datetime.utcnow()
        )
        db.add(db_job)
        db.commit()
        db.refresh(db_job)

        process_video.delay(job_id)

        return db_job

    except Exception as e:
        if 'upload_dir' in locals():
            shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/videos/{job_id}", response_model=VideoJobResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the status of a video processing job.
    No API key required.
    """
    job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/videos/{job_id}/clips", response_model=list[ClipInfo])
async def list_clips(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    List all generated clips for a job.
    No API key required.
    """
    job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    clips_dir = settings.PROCESSED_DIR / job_id
    if not clips_dir.exists():
        return []

    clips = []
    for clip_file in clips_dir.glob("*.mp4"):
        filename = clip_file.name
        rating_str = filename.split("_")[0]
        title = " ".join(filename.split("_")[1:]).rsplit(".", 1)[0]
        clips.append(ClipInfo(
            title=title,
            rating=float(rating_str),
            duration=0.0,
            filename=filename
        ))

    return sorted(clips, key=lambda x: x.rating, reverse=True)


@app.get("/videos/{job_id}/clips/{clip_filename}")
async def download_clip(
    job_id: str,
    clip_filename: str,
    db: Session = Depends(get_db)
):
    """
    Download a specific clip.
    No API key required.
    """
    job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    clip_path = settings.PROCESSED_DIR / job_id / clip_filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    return FileResponse(
        path=clip_path,
        media_type="video/mp4",
        filename=clip_filename
    )


@app.get("/videos/{job_id}/transcript", response_model=FullTranscript)
async def get_full_transcript(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the full transcript for a video.
    No API key required.
    """
    job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    transcript_path = settings.PROCESSED_DIR / job_id / "transcript.json"
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading transcript: {str(e)}")


@app.get("/videos/{job_id}/clips/{clip_filename}/transcript")
async def get_clip_transcript(
    job_id: str,
    clip_filename: str,
    db: Session = Depends(get_db)
):
    """
    Get the transcript for a specific clip.
    No API key required.
    """
    job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    transcript_filename = clip_filename.rsplit(".", 1)[0] + ".txt"
    transcript_path = settings.PROCESSED_DIR / job_id / transcript_filename

    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Clip transcript not found")

    try:
        return FileResponse(
            path=transcript_path,
            media_type="text/plain",
            filename=transcript_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading transcript: {str(e)}")
