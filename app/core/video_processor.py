"""
Video content processing module that automatically extracts, transcribes,
and segments video content into short, engaging clips.

This module uses Whisper for transcription, the Grok API for content analysis,
and various video processing libraries to create subtitled clips.
"""

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import List, Dict, Optional, Callable, Iterator

import numpy as np
import requests
import torch
import whisper
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, VideoClip, CompositeVideoClip
from tqdm import tqdm


@dataclass
class Word:
    """Represents a single word with its timing information."""
    text: str
    start: float
    end: float


@dataclass
class Segment:
    """Represents a video segment with metadata."""
    start: float
    end: float
    title: str
    rating: float
    words: List[Word]
    text: str


class SubtitleFormatter:
    """Handles the creation and formatting of subtitle frames for video clips."""
    
    DEFAULT_FONT = "Arial.ttf"
    
    def __init__(self, video_width: int, video_height: int, font_path: Optional[str] = None):
        self.video_width = video_width
        self.video_height = video_height
        self.font_path = self._get_valid_font_path(font_path)
        self.base_font_size = max(int(video_width * 0.045), 16)
        self.font = self._load_font()
        
        # Styling configuration
        self.text_color = (255, 255, 255, 255)
        self.highlight_color = (255, 255, 0, 255)
        self.stroke_color = (0, 0, 0, 255)
        self.bottom_margin = int(video_height * 0.1)
        self.max_width = int(video_width * 0.85)
        self.subtitle_height = int(video_height * 0.2)

    def _get_valid_font_path(self, font_path: Optional[str]) -> str:
        """Returns a valid font path, falling back to defaults if necessary."""
        paths_to_try = [
            font_path,
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            self.DEFAULT_FONT
        ]
        for path in paths_to_try:
            if path and Path(path).exists():
                return path
        return self.DEFAULT_FONT

    def _load_font(self) -> ImageFont.FreeTypeFont:
        """Loads and returns the font, falling back to default if necessary."""
        try:
            return ImageFont.truetype(self.font_path, self.base_font_size)
        except OSError:
            return ImageFont.load_default()

    def split_phrases(self, words: List[Word], current_time: float) -> List[List[Word]]:
        """Groups words into visually appropriate phrases for display."""
        if not words:
            return []

        phrases = []
        current_phrase = []
        current_width = 0
        
        for word in words:
            word_text = f"{word.text} "
            width = self.font.getlength(word_text)
            
            if current_width + width > self.max_width and current_phrase:
                if current_phrase[-1].end >= current_time:
                    phrases.append(current_phrase)
                current_phrase = []
                current_width = 0
            
            current_phrase.append(word)
            current_width += width
            
            if word.text.rstrip()[-1] in ".!?,":
                if current_phrase[-1].end >= current_time:
                    phrases.append(current_phrase)
                current_phrase = []
                current_width = 0
        
        if current_phrase and current_phrase[-1].end >= current_time:
            phrases.append(current_phrase)
        
        return phrases

    def create_subtitle_frame(self, words: List[Word], current_time: float) -> np.ndarray:
        """Creates a single subtitle frame with the current word highlighted."""
        img = Image.new('RGBA', (self.video_width, self.subtitle_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        phrases = self.split_phrases(words, current_time)
        active_phrase = next((phrase for phrase in phrases 
                            if phrase[0].start <= current_time <= phrase[-1].end), None)
        
        if active_phrase:
            self._draw_phrase(draw, active_phrase, current_time)
        
        return np.array(img)

    def _draw_phrase(self, draw: ImageDraw.Draw, phrase: List[Word], current_time: float):
        """Draws a single phrase with appropriate highlighting and formatting."""
        text_str = " ".join(word.text for word in phrase)
        text_width = draw.textlength(text_str, font=self.font)
        x_pos = (self.video_width - text_width) // 2
        y_pos = self.subtitle_height - self.base_font_size - 20
        
        # Draw background
        padding = 10
        bg_bbox = [
            x_pos - padding,
            y_pos - padding,
            x_pos + text_width + padding,
            y_pos + self.base_font_size + padding
        ]
        draw.rectangle(bg_bbox, fill=(0, 0, 0, 128))
        
        # Draw text outline
        for offset_x, offset_y in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            draw.text((x_pos + offset_x, y_pos + offset_y), 
                     text_str, font=self.font, fill=self.stroke_color)
        
        # Draw words with highlighting
        x_cursor = x_pos
        for word in phrase:
            word_text = f"{word.text} "
            word_width = draw.textlength(word_text, font=self.font)
            color = (self.highlight_color if word.start <= current_time <= word.end 
                    else self.text_color)
            draw.text((x_cursor, y_pos), word_text, font=self.font, fill=color)
            x_cursor += word_width


class TranscriptManager:
    """Manages video transcription using Whisper and handles transcript storage."""
    
    def __init__(self, video_path: str, output_dir: str, model_name: str = "medium"):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.transcript_path = self.output_dir / "transcript.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.output_dir.mkdir(exist_ok=True)
        
        try:
            self.model = whisper.load_model(model_name).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    def load_or_transcribe(self) -> Dict:
        """Loads existing transcript or creates new one if needed."""
        if self.transcript_path.exists():
            return self._load_transcript()
        return self._create_transcript()

    def _load_transcript(self) -> Dict:
        """Loads and parses an existing transcript file."""
        with open(self.transcript_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        
        transcript = {'segments': []}
        for seg in saved['segments']:
            words = [
                Word(text=w['text'], start=float(w['start']), end=float(w['end']))
                for w in seg['words']
            ]
            transcript['segments'].append({
                'start': float(seg['start']),
                'end': float(seg['end']),
                'text': seg['text'],
                'words': words
            })
        
        print("Loaded existing transcript from disk.")
        return transcript

    def _create_transcript(self) -> Dict:
        """Creates a new transcript using Whisper."""
        print("Transcribing video...")
        result = self.model.transcribe(
            str(self.video_path),
            language="en",
            word_timestamps=True,
            verbose=False
        )
        
        transcript = self._process_whisper_result(result)
        self._save_transcript(transcript)
        
        return transcript

    def _process_whisper_result(self, result: Dict) -> Dict:
        """Processes raw Whisper output into structured transcript."""
        transcript = {'segments': []}
        for seg in result['segments']:
            words = []
            if 'words' in seg:
                for w in seg['words']:
                    text_val = w.get('text', w.get('word', '')).strip()
                    words.append(Word(
                        text=text_val,
                        start=float(w['start']),
                        end=float(w['end'])
                    ))
            
            transcript['segments'].append({
                'start': float(seg['start']),
                'end': float(seg['end']),
                'text': seg['text'].strip(),
                'words': words
            })
        
        return transcript

    def _save_transcript(self, transcript: Dict):
        """Saves transcript to disk in JSON format."""
        json_transcript = {
            'segments': [{
                'start': s['start'],
                'end': s['end'],
                'text': s['text'],
                'words': [{
                    'text': w.text,
                    'start': w.start,
                    'end': w.end
                } for w in s['words']]
            } for s in transcript['segments']]
        }
        
        with open(self.transcript_path, 'w', encoding='utf-8') as f:
            json.dump(json_transcript, f, indent=2, ensure_ascii=False)
        
        print("Transcript saved to disk.")


class SegmentAnalyzer:
    """Analyzes transcript chunks using the Grok API to identify meaningful segments."""

    SYSTEM_PROMPT = """# Video Content Segment Analyzer

You are an expert content analyst specializing in video editing and audience engagement. Your task is to 
analyze video transcripts and identify high-quality, self-contained segments that would be engaging as 
standalone clips. Look for moments that are compelling, memorable, or provide value to viewers.

## What Makes a Good Segment
- Natural beginning and end points
- Contains a complete thought, story, or idea
- Engaging and meaningful on its own
- Clear focus or purpose
- High audience retention potential
- Could work as a short-form video

## Segment Requirements
- Length: 30-180s (optimal 45-90s for social sharing)
- Must be self-contained with clear context
- Should have a clear hook or point of interest
- Avoid cutting mid-sentence or during key context

## Content Evaluation Criteria
Rate segments based on:
- Engagement value (how likely viewers are to watch fully)
- Standalone quality (how well it works without extra context)
- Production value (audio clarity, speaking pace, energy)
- Share-worthiness (how likely viewers are to share)
- Emotional impact or intellectual value

## Output Format
Return output ONLY as a JSON array of segments. Each segment MUST include:
- "start": float (seconds from chunk start)
- "end": float (seconds from chunk start)
- "title": string (concise, attention-grabbing title)
- "rating": float (0.0-1.0, based on evaluation criteria)

Example output:
[
  {
    "start": 45.2,
    "end": 115.8,
    "title": "The Shocking Truth About Deep Sea Creatures",
    "rating": 0.92
  },
  {
    "start": 180.5,
    "end": 245.0,
    "title": "How This Simple Trick Changed Everything",
    "rating": 0.85
  }
]

IMPORTANT: Return ONLY the JSON array, no other text or explanation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def analyze_chunk(self, chunk_text: str, chunk_start: float) -> List[Dict]:
        """Analyzes a transcript chunk and returns valid segments."""
        if not chunk_text.strip():
            print("Empty chunk, skipping analysis.")
            return []

        print(f"Analyzing chunk from {chunk_start:.1f}s...")
        try:
            response = self._make_api_request(chunk_text)
            segments = self._parse_api_response(response)
            return self._validate_segments(segments, chunk_start)
        except Exception as e:
            print(f"Error analyzing chunk: {e}")
            return []

    def _make_api_request(self, chunk_text: str) -> Dict:
        """Makes request to Grok API and returns response."""
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=self.headers,
            json={
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze this transcript chunk and return ONLY the JSON array:\n\n{chunk_text}"}
                ],
                "model": "grok-beta",
                "stream": False,
                "temperature": 0.1
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def _parse_api_response(self, response: Dict) -> List[Dict]:
        """Extracts and parses JSON content from API response."""
        content = response['choices'][0]['message']['content']
        
        # Extract JSON from code blocks if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        try:
            segments = json.loads(content.strip())
            if not isinstance(segments, list):
                raise ValueError("API response is not a list")
            return segments
        except json.JSONDecodeError as e:
            print(f"Failed to parse API response: {e}")
            return []

    def _validate_segments(self, segments: List[Dict], chunk_start: float) -> List[Dict]:
        """Validates segments against required criteria."""
        valid_segments = []
        for seg in segments:
            if not all(k in seg for k in ['start', 'end', 'title', 'rating']):
                continue
                
            try:
                duration = float(seg['end']) - float(seg['start'])
                rating = float(seg['rating'])
            except (ValueError, TypeError):
                continue
                
            if 30 <= duration <= 180 and 0.0 <= rating <= 1.0:
                seg['start'] = chunk_start + float(seg['start'])
                seg['end'] = chunk_start + float(seg['end'])
                valid_segments.append(seg)
        
        return valid_segments

class ClipExtractor:
    """Handles video clip extraction and subtitle overlay."""
    
    def __init__(self, video_path: str, output_dir: str, font_path: Optional[str] = None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.font_path = font_path
        self.output_dir.mkdir(exist_ok=True)

    def extract_clip(self, segment: Segment):
        """Extracts a video clip with subtitles and saves related metadata."""
        output_paths = self._generate_output_paths(segment)
        
        self._save_metadata(segment, output_paths['text'])
        self._create_video_clip(segment, output_paths['video'])

    def _generate_output_paths(self, segment: Segment) -> Dict[str, Path]:
        """Generates standardized output paths for clip files."""
        rating_str = f"{segment.rating:.2f}"
        safe_title = "".join(c for c in segment.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_') or "untitled_segment"
        
        return {
            'video': self.output_dir / f"{rating_str}_{safe_title}.mp4",
            'text': self.output_dir / f"{rating_str}_{safe_title}.txt"
        }

    def _save_metadata(self, segment: Segment, output_path: Path):
        """Saves segment metadata and transcript to text file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Title: {segment.title}\n")
            f.write(f"Rating: {segment.rating:.2f}\n")
            f.write(f"Time: {segment.start:.1f}s - {segment.end:.1f}s\n\n")
            f.write("Transcript:\n")
            for word in segment.words:
                f.write(f"[{word.start:.1f}s - {word.end:.1f}s] {word.text} ")
            f.write("\n\nPlain text:\n")
            f.write(segment.text)

    def _create_video_clip(self, segment: Segment, output_path: Path):
        """Creates and saves video clip with subtitles."""
        print(f"Extracting clip '{segment.title}' ({segment.start:.1f}s - {segment.end:.1f}s)")
        
        with VideoFileClip(str(self.video_path)) as video:
            video_segment = video.subclip(segment.start, segment.end)
            subtitle_clip = self._create_subtitle_clip(segment, video_segment)
            
            final_clip = CompositeVideoClip(
                [video_segment, subtitle_clip],
                size=(video_segment.w, video_segment.h)
            )
            
            final_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                bitrate="8000k",
                audio_bitrate="320k",
                fps=video_segment.fps,
                preset="medium",
                logger=None
            )
            
            final_clip.close()
            subtitle_clip.close()

    def _create_subtitle_clip(self, segment: Segment, video: VideoClip) -> VideoClip:
        """Creates subtitle overlay for video clip."""
        formatter = SubtitleFormatter(video.w, video.h, self.font_path)
        
        def make_frame(t):
            current_time = t + segment.start
            frame = formatter.create_subtitle_frame(segment.words, current_time)
            
            if frame.shape[2] == 4:
                # Convert RGBA to RGB by compositing on black background
                rgb = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                alpha = frame[:, :, 3:4] / 255.0
                rgb = (frame[:, :, :3] * alpha + rgb * (1 - alpha)).astype(np.uint8)
                return rgb
            return frame

        return VideoClip(make_frame, duration=segment.end - segment.start).set_position(('center', 'bottom'))


class GrokLectureClipper:
    """
    Main orchestrator for video processing:
    - Transcribes video content
    - Analyzes content for meaningful segments
    - Extracts segments as standalone clips with subtitles
    """
    
    def __init__(
        self,
        video_path: str,
        api_key: str,
        output_dir: str,
        model_name: str = "medium",
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        self.output_dir = Path(output_dir)
        self.progress_callback = progress_callback or (lambda x: None)
        
        self.transcript_mgr = TranscriptManager(video_path, output_dir, model_name)
        self.analyzer = SegmentAnalyzer(api_key)
        self.clip_extractor = ClipExtractor(video_path, output_dir)

    def run(self):
        """Executes the full video processing pipeline with progress updates."""
        try:
            self.progress_callback(0.0)
            print("Starting video processing...")
            
            # Transcribe
            transcript = self.transcript_mgr.load_or_transcribe()
            self.progress_callback(0.3)
            
            with VideoFileClip(str(self.video_path)) as video:
                duration = video.duration
            
            # Process chunks
            total_chunks = ceil(duration / 1500)
            for i, (chunk_start, chunk_end) in enumerate(self._get_chunks(duration)):
                self._process_chunk(transcript, chunk_start, chunk_end)
                self.progress_callback(0.3 + (0.7 * (i + 1) / total_chunks))
            
            self.progress_callback(1.0)
            print("Processing complete.")
            
        except Exception as e:
            print(f"Error in processing: {e}")
            raise

    def _get_chunks(self, duration: float, chunk_size: float = 1500) -> Iterator[tuple[float, float]]:
        """Yields time chunks for processing."""
        start = 0
        while start < duration:
            end = min(start + chunk_size, duration)
            yield (start, end)
            start = end

    def _get_chunk_text(self, transcript: Dict, start_time: float, end_time: float) -> str:
        """Extracts transcript text for a specific time chunk."""
        chunk_lines = []
        for seg in transcript['segments']:
            if seg['start'] >= start_time and seg['end'] <= end_time:
                rel_start = seg['start'] - start_time
                rel_end = seg['end'] - start_time
                chunk_lines.append(f"[{rel_start:.1f}s - {rel_end:.1f}s] {seg['text']}")
        return "\n".join(chunk_lines)

    def _process_chunk(self, transcript: Dict, chunk_start: float, chunk_end: float):
        """Processes a single chunk of the video."""
        print(f"\nProcessing chunk {chunk_start:.1f}s to {chunk_end:.1f}s...")
        
        chunk_text = self._get_chunk_text(transcript, chunk_start, chunk_end)
        segments = self.analyzer.analyze_chunk(chunk_text, chunk_start)
        
        if not segments:
            print("No valid segments found in this chunk.")
            return
            
        for seg_dict in segments:
            segment = self._create_segment(transcript, seg_dict)
            if segment:
                try:
                    self.clip_extractor.extract_clip(segment)
                except Exception as e:
                    print(f"Error extracting clip: {e}")

    def _create_segment(self, transcript: Dict, seg_dict: Dict) -> Optional[Segment]:
        """Creates a Segment object from transcript data and segment metadata."""
        clip_words = []
        clip_text = []
        
        for t_seg in transcript['segments']:
            if ((seg_dict['start'] <= t_seg['start'] < seg_dict['end']) or 
                (seg_dict['start'] < t_seg['end'] <= seg_dict['end'])):
                clip_text.append(t_seg['text'])
                for word in t_seg['words']:
                    if seg_dict['start'] <= word.start < seg_dict['end']:
                        clip_words.append(word)
        
        if not clip_words:
            return None
            
        return Segment(
            start=seg_dict['start'],
            end=seg_dict['end'],
            title=seg_dict['title'],
            rating=seg_dict['rating'],
            words=clip_words,
            text=' '.join(clip_text)
        )