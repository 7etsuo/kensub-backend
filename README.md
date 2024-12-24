# Kensub Video Content Analysis and Clipping Service

An API service that automatically processes videos to create engaging clips with subtitles. It uses Whisper for transcription and the Grok API for intelligent content analysis.

## Features

- Automatic video transcription using OpenAI's Whisper
- Intelligent content segmentation using Grok API
- Automatic subtitle generation and overlay
- REST API interface for video processing
- Progress tracking and job management
- Multi-user support
- Clip and transcript downloads

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- NVIDIA GPU (optional, for faster processing)

### GPU Support (Optional)

If you want to use GPU acceleration for faster transcription:

For Debian:
```bash
# Add NVIDIA repository & GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/debian11/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd video-content-service
```

2. Create a `.env` file:
```env
GROK_API_KEY=your_grok_api_key
DATABASE_URL=postgresql://user:password@postgres/videodb
REDIS_URL=redis://redis:6379/0
```

3. Start the services:
```bash
docker-compose up --build
```

## API Endpoints

### Upload Video
```http
POST /videos/
```
Upload a video file for processing. Returns a job ID.

### Check Job Status
```http
GET /videos/{job_id}
```
Get the current status of a processing job.

### List Clips
```http
GET /videos/{job_id}/clips
```
Get a list of generated clips for a job.

### Download Clip
```http
GET /videos/{job_id}/clips/{clip_filename}
```
Download a specific clip.

### Get Full Transcript
```http
GET /videos/{job_id}/transcript
```
Get the complete transcript of the video.

### Get Clip Transcript
```http
GET /videos/{job_id}/clips/{clip_filename}/transcript
```
Get the transcript for a specific clip.

## Usage Example

```python
import requests

# Set your API key
headers = {'X-API-Key': 'your_api_key'}

# Upload a video
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/videos/', 
        files={'file': f},
        headers=headers
    )
job_id = response.json()['id']

# Check job status
status = requests.get(
    f'http://localhost:8000/videos/{job_id}', 
    headers=headers
).json()

# Get full transcript
transcript = requests.get(
    f'http://localhost:8000/videos/{job_id}/transcript', 
    headers=headers
).json()

# List clips
clips = requests.get(
    f'http://localhost:8000/videos/{job_id}/clips', 
    headers=headers
).json()

# Download clip
clip_filename = clips[0]['filename']
clip = requests.get(
    f'http://localhost:8000/videos/{job_id}/clips/{clip_filename}', 
    headers=headers
).content

# Get clip transcript
transcript = requests.get(
    f'http://localhost:8000/videos/{job_id}/clips/{clip_filename}/transcript', 
    headers=headers
).text
```

## Project Structure
```
app/
├── __init__.py
├── main.py           # FastAPI endpoints
├── config.py         # Configuration settings
├── models.py         # Database models
├── database.py       # Database connection
├── schemas.py        # Pydantic models
├── tasks.py         # Celery tasks
└── core/
    └── video_processor.py  # Core processing logic
```

## Configuration

The service can be configured through environment variables:

- `GROK_API_KEY`: Your Grok API key
- `MAX_CONCURRENT_JOBS`: Maximum number of concurrent processing jobs (default: 3)
- `MAX_UPLOAD_SIZE`: Maximum video file size in bytes (default: 1GB)
- `WHISPER_MODEL`: Whisper model to use (default: "medium")

## Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
uvicorn app.main:app --reload
```