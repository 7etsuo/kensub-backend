from celery import Celery
from celery.signals import worker_ready
from datetime import datetime
import redis
from sqlalchemy.exc import ProgrammingError

from .config import settings
from .database import SessionLocal
from .models import VideoJob
from .core.video_processor import GrokLectureClipper

celery_app = Celery('video_processor', broker=settings.REDIS_URL)

redis_client = redis.from_url(settings.REDIS_URL)


@worker_ready.connect
def at_start(sender, **kwargs):
    """Reset any stale jobs when worker starts."""
    db = SessionLocal()
    try:
        stale_jobs = db.query(VideoJob).filter(
            VideoJob.status.in_(['processing', 'pending'])
        ).all()
        for job in stale_jobs:
            job.status = 'failed'
            job.error_message = 'Job reset due to server restart'
        db.commit()
    except ProgrammingError as e:
        print(f"Skipping stale job reset because the table doesn't exist yet. Error: {e}")
    finally:
        db.close()


def can_start_job() -> bool:
    """Check if we can start a new job based on concurrent limits."""
    current_jobs = int(redis_client.get('current_jobs') or 0)
    if current_jobs >= settings.MAX_CONCURRENT_JOBS:
        return False
    redis_client.incr('current_jobs')
    return True


def finish_job():
    """Decrement the job counter."""
    redis_client.decr('current_jobs')


@celery_app.task(bind=True)
def process_video(self, job_id: str):
    """Celery task for processing a video."""
    if not can_start_job():
        # Requeue the task if we're at capacity
        self.retry(countdown=60)
        return

    db = SessionLocal()
    try:
        job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
        if not job:
            return

        job.status = "processing"
        db.commit()

        def progress_callback(percent: float):
            job.progress = percent
            db.commit()
            self.update_state(state='PROGRESS', meta={'progress': percent})

        # Process video
        clipper = GrokLectureClipper(
            video_path=str(settings.UPLOAD_DIR / job_id / "video.mp4"),
            api_key=settings.GROK_API_KEY,
            output_dir=str(settings.PROCESSED_DIR / job_id),
            progress_callback=progress_callback
        )
        clipper.run()

        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.progress = 100.0
        db.commit()

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise
    finally:
        finish_job()
        db.close()
